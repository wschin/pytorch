#include <torch/csrc/jit/passes/onnx.h>

#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/symbolic.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/shape_type_inference.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/utils/pybind.h>
#include <sstream>
#include <unordered_map>
#include <tuple>

namespace torch {
namespace jit {

void removePrintOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto b : it->blocks()) {
      removePrintOps(b);
    }
    if (it->kind() == prim::Print || it->kind() == aten::warn) {
      for (size_t i = 0; i < it->inputs().size();) {
        auto input = it->inputs().at(i);
        // only handling constants bc of potential side effects
        if (input->uses().size() == 1 &&
            input->node()->kind() == prim::Constant) {
          it->removeInput(i);
          input->node()->destroy();
        } else {
          ++i;
        }
      }
      it.destroyCurrent();
    }
  }
}

void RemovePrintOps(std::shared_ptr<Graph>& graph) {
  removePrintOps(graph->block());
}

void checkONNXCompatibility(const c10::FunctionSchema& schema) {
  // in ONNX, all inputs are tensors, no support for tensor list
  // so at most one input tensor list is supported
  bool has_tensor_list = false;
  const auto& args = schema.arguments();
  for (const auto& arg : args) {
    if (arg.name() == "_caffe2_preallocated_outputs") {
      continue;
    }
    auto type = arg.type();
    if (type->kind() == TypeKind::OptionalType) {
      type = reinterpret_cast<OptionalType*>(type.get())->getElementType();
      AT_ASSERT(type->kind() != TypeKind::OptionalType);
    }
    if (type->kind() == TypeKind::ListType) {
      const auto& elem_type =
          reinterpret_cast<ListType*>(type.get())->getElementType();
      if (elem_type->isSubtypeOf(TensorType::get())) {
        AT_ASSERTM(
            !has_tensor_list,
            "ONNX export supports at most one TensorList as input.");
        has_tensor_list = true;
      }
    }
  }
}

void preprocessCaffe2Ops(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto b : it->blocks()) {
      preprocessCaffe2Ops(b);
    }
    if (it->kind().is_caffe2()) {
      const auto& schema = it->schema();
      checkONNXCompatibility(schema);
      std::vector<Value*> origin_inputs;
      for (Value* v : it->inputs()) {
        origin_inputs.push_back(v);
      }
      it->removeAllInputs();
      const auto& args = schema.arguments();
      size_t origin_inputs_index = 0;
      for (const auto& arg : args) {
        auto type = arg.type();
        AT_ASSERT(origin_inputs_index < origin_inputs.size());
        const auto& origin_input = origin_inputs[origin_inputs_index++];
        if (type->kind() == TypeKind::OptionalType) {
          type = reinterpret_cast<OptionalType*>(type.get())->getElementType();
          if (origin_input->mustBeNone()) {
            continue;
          } else {
            // recursive optional type is not supported
            AT_ASSERT(type->kind() != TypeKind::OptionalType);
          }
        }
        if (type->isSubtypeOf(TensorType::get())) {
          it->addInput(origin_input);
        } else if (
            type->kind() == TypeKind::BoolType ||
            type->kind() == TypeKind::IntType) {
          const auto* constant_node = origin_input->node();
          AT_ASSERT(constant_node->kind() == prim::Constant);
          it->i_(Symbol::attr(arg.name()), constant_node->i(attr::value));
        } else if (type->kind() == TypeKind::FloatType) {
          const auto* constant_node = origin_input->node();
          AT_ASSERT(constant_node->kind() == prim::Constant);
          it->f_(Symbol::attr(arg.name()), constant_node->f(attr::value));
        } else if (type->kind() == TypeKind::StringType) {
          const auto* constant_node = origin_input->node();
          AT_ASSERT(constant_node->kind() == prim::Constant);
          it->s_(Symbol::attr(arg.name()), constant_node->s(attr::value));
        } else if (type->kind() == TypeKind::ListType) {
          const auto& list_node = origin_input->node();
          const auto& elem_type = type->castRaw<ListType>()->getElementType();
          AT_ASSERT(
              list_node->kind() == prim::ListConstruct ||
              list_node->kind() == prim::Constant);
          if (elem_type->isSubtypeOf(TensorType::get())) {
            AT_ASSERT(list_node->kind(), prim::ListConstruct);
            const auto& tensor_list = origin_input->node()->inputs();
            for (const auto& t : tensor_list) {
              it->addInput(t);
            }
          } else if (elem_type->kind() == TypeKind::FloatType) {
            std::vector<double> values;
            if (list_node->kind() == prim::ListConstruct) {
              for (const auto* elem_input : list_node->inputs()) {
                const auto* constant_node = elem_input->node();
                AT_ASSERT(constant_node->kind() == prim::Constant);
                values.push_back(constant_node->f(attr::value));
              }
            } else { // is a constant list
              values = list_node->fs(attr::value);
            }
            it->fs_(Symbol::attr(arg.name()), values);
          } else {
            throw std::runtime_error(
                "Unhandled scalar arg: " + arg.name() +
                ", type: " + c10::typeKindToString(elem_type->kind()));
          }
        } else {
          throw std::runtime_error(
              "Unsupported input type of arg " + arg.name() +
              " in Caffe2 operator: " + c10::typeKindToString(type->kind()));
        }
      }
    }
  }
  EliminateDeadCode(
      block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

void PreprocessCaffe2Ops(std::shared_ptr<Graph>& graph) {
  preprocessCaffe2Ops(graph->block());
}

// Transform PythonOps into Nodes that match ONNX semantics.
std::shared_ptr<Graph> ToONNX(
    std::shared_ptr<Graph>& graph,
    ::torch::onnx::OperatorExportTypes operator_export_type) {
  auto constant_value_map = ConstantValueMap::getInstance();
  ConstantValueMap::ClearMaps();
  auto new_graph = std::make_shared<Graph>(graph->current_scope());
  std::unordered_map<Value*, Value*> env;
  BlockToONNX(graph->block(), new_graph->block(), operator_export_type, env);
  return new_graph;
}

void BlockToONNX(
    Block* old_block,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*> env) {
  torch::autograd::SymbolicContext ctx{};
  ctx.block = new_block;

  GRAPH_DEBUG(
      "BlockToONNX: graph of old block: ",
      old_block->owningGraph()->toString());

  // Initialize context and environment
  for (auto input : old_block->inputs()) {
    auto n = ctx.block->addInput()->copyMetadata(input);
    env[input] = n;
  }

  // Finally, visit all nodes in the graph
  for (auto node : old_block->nodes()) {
    NodeToONNX(node, ctx.block, operator_export_type, env);
  }
  for (auto output : old_block->outputs()) {
    ctx.block->registerOutput(env.at(output));
  }

  // Run dce to clean-up unused functional and inplace ops.
  EliminateDeadCode(
      ctx.block,
      true,
      DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

void NodeToONNX(
    Node* old_node,
    Block* new_block,
    ::torch::onnx::OperatorExportTypes operator_export_type,
    std::unordered_map<Value*, Value*>& env) {
  py::object onnx = py::module::import("torch.onnx");
  py::object onnx_symbolic = py::module::import("torch.onnx.symbolic_helper");
  py::object onnx_registry = py::module::import("torch.onnx.symbolic_registry");

  // Setup all the lambda helper functions.

  // Returns a node that n maps to in the new graph
  auto envFn = [&env](Value* n) -> Value* {
    auto it = env.find(n);
    TORCH_CHECK(it != env.end(), "Dangling node reference");
    TORCH_CHECK(it->second, "Unused node was subsequently used");
    return it->second;
  };

  // Put the new outputs in our environment map, and copy the type from the
  // input graph if they were not set by the symbolic. This is called only
  // with results of symbolic call (not for nodes that are just cloned).
  auto setOutputs = [&](const std::string& op_name,
                        Node* node,
                        const value_list& outputs) {
    auto old_outputs = node->outputs();
    // Count all outputs, excluding Handles
    auto num_old_outputs = old_outputs.size();
    if (outputs.size() != num_old_outputs) {
      std::ostringstream ss;
      ss << "symbolic for " << op_name
         << " produced an incorrect number of outputs (expected ";
      ss << num_old_outputs << ", but got " << outputs.size() << ")";
      throw std::runtime_error(ss.str());
    }
    for (size_t i = 0; i < num_old_outputs; ++i) {
      auto old = old_outputs[i];
      if (outputs[i]) {
        // Allow symbolic() to skip specifying the type of the return node.
        // Unfortunately, they are on the hook for all internal nodes
        // (though in practice, the types are not computed.)
        //
        // If onnx shape inference is turned on, the new outputs will have
        // types inferred, and they will be merged with the old types.
        outputs[i]->setType(MergeInferredType(old->type(), outputs[i]->type()));

        // Copy over source location and scope information to all nodes
        // created by the symbolic
        outputs[i]->node()->setSourceRange(node->sourceRange());
        outputs[i]->node()->setScope(node->scope());
        env[old] = outputs[i];
      } else {
        // Null output means that the ONNX op doesn't have outputs corresponding
        // to certain PyTorch outputs
        env[old] = nullptr;
        if (!old->uses().empty()) {
          std::ostringstream ss;
          ss << "symbolic for " << op_name << " returned None for the output "
             << i;
          ss << " (indicating conversion for that particular output is not supported), ";
          ss << "but the network uses this output later";
          // TODO: Say what actually used it
          throw std::runtime_error(ss.str());
        }
      }
    }
  };

  // Clone the node and add it to the new graph
  auto cloneNode = [&](Node* node) {
    auto n_ = new_block->appendNode(
        new_block->owningGraph()->createClone(node, envFn));
    for (size_t i = 0; i < node->outputs().size(); i++) {
      // n_->outputs()[i]->setType(node->outputs()[i]->type());
      env[node->output(i)] = n_->output(i);
    }
  };

  auto clonePythonOp = [&](ConcretePythonOp* node) {
    std::cout << "[onnx.cpp] Python node: " << *node << std::endl;
    py::object _training_mode = onnx_symbolic.attr("_training_mode");
    int64_t training_mode = _training_mode.cast<int64_t>();
    auto n_ = new_block->appendNode(
        new_block->owningGraph()->createClone(node, envFn));
    for (size_t i = 0; i < node->outputs().size(); i++) {
      // n_->outputs()[i]->setType(node->outputs()[i]->type());
      env[node->output(i)] = n_->output(i);
    }
    n_->s_(Symbol::attr("name"), node->name());
    n_->i_(Symbol::attr("training_mode"), training_mode);

    // Attributes for tensor inputs.
    std::vector<int64_t> input_tensor_types;
    std::vector<int64_t> input_tensor_requires_grads;

    // Attributes for Python's int inputs. Its value can be arbitrarily large, but
    // we assume it can be stored as int64_t in C++.
    std::vector<int64_t> input_int_scalars;
    // input_int64_scalar_positions[i] is the position index of input_int64_scalars[i]
    // when calling the original autograd.Function.apply.
    std::vector<int64_t> input_int_scalar_positions;

    // Attributes for Python's double inputs.
    std::vector<double> input_float_scalars;
    std::vector<int64_t> input_float_scalar_positions;

    std::vector<int64_t> input_int_tuples;
    std::vector<int64_t> input_int_tuple_positions;
    std::vector<int64_t> input_int_tuple_begins;

    std::vector<double> input_float_tuples;
    std::vector<int64_t> input_float_tuple_positions;
    std::vector<int64_t> input_float_tuple_begins;

    std::vector<int64_t> input_pointer_scalars; 
    std::vector<int64_t> input_pointer_scalar_positions;

    // "apply_index=i" means the i-th input argument of apply(...).
    // "scalar_index" is position index to scalar arguments of apply(...).
    // For example, "index=0" means the 1st scalar argument.
    // Note that tensors are not indexed by this "scalar_index".
    auto process_scalar = [&] (const size_t apply_index, const size_t scalar_index) {
      auto& arg = node->scalar_args.at(scalar_index);
      auto arg_raw = arg.get();
      auto arg_handle = py::handle(arg.get());

      std::cout << "[onnx.cc,process_scalar] Process " << std::endl;
      py::print(arg_handle);

      // Store attributes of this scalar.

      input_tensor_requires_grads.push_back(0);
      if (py::isinstance<py::int_>(arg_handle)) {
        // Case 1: See a Python int.
        input_int_scalar_positions.push_back(apply_index);
        const int64_t value = py::cast<int64_t>(arg_handle);
        input_int_scalars.push_back(value);
      } else if (py::isinstance<py::float_>(arg_handle)) {
        // Case 2: See a Python float.
        input_float_scalar_positions.push_back(apply_index);
        const double value = py::cast<double>(arg_handle);
        input_float_scalars.push_back(value);
      } else if (py::isinstance<py::tuple>(arg_handle)) {
        // Case 3: See a Python tuple.
        // Set tuple-wise attributes.
        py::handle item = PyTuple_GET_ITEM(arg_raw, 0);
        if (py::isinstance<py::int_>(item)) {
          input_int_tuple_positions.push_back(apply_index);
          input_int_tuple_begins.push_back(input_int_tuples.size());
        } else if (py::isinstance<py::float_>(item)) {
          input_float_tuple_positions.push_back(apply_index);
          input_float_tuple_begins.push_back(input_float_tuples.size());
        } else {
          std::ostringstream ss;
          ss << "Error casting " << 0 << "th input element in Python tuple "
             << "when processing node " << node->name() << ". "
             << "Only float and int are supported." ;
          throw std::runtime_error(ss.str());
        }

        // Store tuple elements.
        // All same-type tuples' elements are stored in a flattened list.
        // Get the length of this tuple.
        const auto n_elements = static_cast<size_t>(PyTuple_GET_SIZE(arg_raw));
        for (size_t i = 0; i < n_elements; ++i) {
          py::handle item = PyTuple_GET_ITEM(arg_raw, i);
          if (py::isinstance<py::int_>(item)) {
            const int64_t value = py::cast<int64_t>(item);
            input_int_tuples.push_back(value);
          } else if (py::isinstance<py::float_>(item)) {
            const double value = py::cast<double>(item);
            input_float_tuples.push_back(value);
          } else {
            std::ostringstream ss;
            ss << "Error casting " << i << "th input element in Python tuple "
               << "when processing node " << node->name() << ". "
               << "That tuple is the "
               << apply_index
               << "th input argument of Python function. "
               << "Only float and int are supported. ";
            py::print(arg_handle);
            py::print(item);
            throw std::runtime_error(ss.str());
          }
        }
      } else {
        input_pointer_scalar_positions.push_back(apply_index);
        std::cout << "[onnx.cpp] raw obj: " << std::endl;
        PyObject_Print(arg_raw, stdout, 0);
        Py_INCREF(arg_raw);
        Py_INCREF(arg_raw);
        Py_INCREF(arg_raw);
        std::cout << std::endl;
        input_pointer_scalars.push_back((int64_t)arg_raw);
        //std::ostringstream ss;
        //ss << "Error casting " << apply_index << "th input argument of Python function "
        //   << "when processing node " << node->name() << ". "
        //   << "Only float, int, and tuple are supported." ;
        //py::print(arg_handle);
        //throw std::runtime_error(ss.str());
      }
    };

    // "apply_index=i" means the i-th input argument of apply(...).
    // "tensor_index" is position index to tensor arguments of apply(...).
    // For example, "index=0" means the 1st tensor argument.
    // Note that scalars are not indexed by this "tensor_index".
    auto process_tensor = [&] (const size_t apply_index, const size_t tensor_index) {
      const auto tensor = n_->inputs().at(tensor_index);
      const c10::TensorTypePtr& tensor_type = tensor->type()->cast<TensorType>();
      const int64_t onnx_type = ATenTypeToOnnxType(tensor_type->scalarType().value());
      // Store attributes of this tensor.
      input_tensor_types.push_back(onnx_type);
      input_tensor_requires_grads.push_back(tensor->requires_grad());
    };

    // Encode inputs of PythonOp.
    size_t visited_tensor_count = 0;
    size_t visited_scalar_count = 0;
    for (size_t i = 0; i < node->cconv.size(); ++i) {
      char arg_type =  node->cconv.at(i);
      if (arg_type == 'c') {
        // Process visited_scalar_count-th scalar input which is from
        // the i-th input argument of apply(...).
        // This will be a part of attributes in ONNX PythonOp.
        process_scalar(i, visited_scalar_count++);
      } else if (arg_type == 'd') {
        // Process visited_tensor_count-th tensor input which is from
        // the i-th input argument of apply(...).
        // This will be an input to ONNX PythonOp.
        process_tensor(i, visited_tensor_count++);
      } else {
        std::ostringstream ss;
        ss << "Error type " << arg_type << ". Only \'c\' and \'d\' are allowed. ";
        throw std::runtime_error(ss.str());
      }
    }

    std::cout << "[onnx.cpp] Python node " << node->name() << " has "
              << node->inputs().size() << " inputs, "
              << node->outputs().size() << " outputs, "
              << node->cconv << " cconv." << std::endl;
    // Encode outputs of PythonOp. They are assumed to be tensors.
    std::vector<int64_t> output_tensor_types;
    std::vector<int64_t> output_tensor_requires_grads;
    for (const auto o: node->outputs()) {
      const c10::TensorTypePtr& tensor_type = o->type()->cast<TensorType>();
      const int64_t onnx_type = ATenTypeToOnnxType(tensor_type->scalarType().value());
      output_tensor_types.push_back(onnx_type);
      output_tensor_requires_grads.push_back(o->requires_grad());
    }

    // The string which specifies input arguments of 
    n_->s_(Symbol::attr("call_convention"), node->cconv);
    n_->is_(Symbol::attr("input_tensor_types"), input_tensor_types);
    n_->is_(Symbol::attr("input_tensor_requires_grads"), input_tensor_requires_grads);
    n_->is_(Symbol::attr("output_tensor_types"), output_tensor_types);
    n_->is_(Symbol::attr("output_tensor_requires_grads"), output_tensor_requires_grads);

    // Input int scalars.
    if (input_int_scalars.size()) {
      n_->is_(Symbol::attr("input_int_scalars"), input_int_scalars);
      n_->is_(Symbol::attr("input_int_scalar_positions"), input_int_scalar_positions);
    }

    // Input float scalars.
    if (input_float_scalars.size()) {
      n_->fs_(Symbol::attr("input_float_scalars"), input_float_scalars);
      n_->is_(Symbol::attr("input_float_scalar_positions"), input_float_scalar_positions);
    }

    // Input int tuple.
    if (input_int_tuples.size()) {
      n_->is_(Symbol::attr("input_int_tuples"), input_int_tuples);
      n_->is_(Symbol::attr("input_int_tuple_positions"), input_int_tuple_positions);
      n_->is_(Symbol::attr("input_int_tuple_begins"), input_int_tuple_begins);
    }

    // Input double tuple.
    if (input_float_tuples.size()) {
      n_->fs_(Symbol::attr("input_float_tuples"), input_float_tuples);
      n_->is_(Symbol::attr("input_float_tuple_positions"), input_float_tuple_positions);
      n_->is_(Symbol::attr("input_float_tuple_begins"), input_float_tuple_begins);
    }

    if(input_pointer_scalars.size()) {
      n_->is_(Symbol::attr("input_pointer_scalars"), input_pointer_scalars);
      n_->is_(Symbol::attr("input_pointer_scalar_positions"), input_pointer_scalar_positions);
    }
  };

  // Cast output of symbolic() python implementation
  auto processSymbolicOutput = [&](const std::string& op_name,
                                   Node* n,
                                   const py::object& raw_output) {
    if (raw_output.ptr() == Py_None) {
      cloneNode(n);
      return;
    }
    // Cast the outputs back to C++ and put them in the new graph
    std::vector<Value*> outputs;
    try {
      if (py::isinstance<Value>(raw_output)) {
        outputs = value_list{py::cast<Value*>(raw_output)};
      } else {
        outputs = py::cast<std::vector<Value*>>(raw_output);
      }
    } catch (const std::exception& ex) {
      std::ostringstream ss;
      ss << "Error casting results of symbolic for " << op_name
         << ": expected to return list of op nodes, instead received type ''"
         << py::str(raw_output.get_type()) << "': " << py::str(raw_output);
      throw std::runtime_error(ss.str());
    }

    setOutputs(op_name, n, outputs);
  };

  auto callPySymbolicFunction = [&](Node* n) {
    // The idea is delegate as much of the actual argument massaging to
    // Python as possible

    py::tuple py_inputs(n->inputs().size());
    Py_ssize_t input_nr = 0;
    for (auto* input : n->inputs()) {
      py_inputs[input_nr++] = py::cast(envFn(input));
    }

    WithInsertPoint insert_point_guard(new_block);
    WithCurrentScope scope_guard(*new_block->owningGraph(), n->scope());
    py::object raw_output = onnx.attr("_run_symbolic_function")(
        new_block->owningGraph(),
        new_block,
        n,
        py_inputs,
        env,
        operator_export_type);

    // TODO: Assert it's an ATen identifier???
    // (Sometimes it's not...)
    processSymbolicOutput(n->kind().toUnqualString(), n, raw_output);
    GRAPH_DUMP("after process output:", new_block->owningGraph());
  };

  auto callPySymbolicMethod = [&](ConcretePythonOp* op) {
    // Test if there is a symbolic function; bail if there is not
    bool aaa = false;
    while(aaa) {
      aaa = aaa;
    }
    auto pyobj = py::handle(op->pyobj.get());
    auto func = op->autogradFunction();
    if (func) {
      pyobj = func->get();
    }
    if (!py::hasattr(pyobj, "symbolic")) {
      // cloneNode(op);
      clonePythonOp(op);
      return;
    }

    // Prepare args for Python. First one is the graph, and is followed
    // by regular args, with Variables replaced by corresponding nodes.
    Py_ssize_t input_nr = 0;
    py::tuple py_symbolic_args(1 + op->cconv.size());
    py_symbolic_args[input_nr++] = py::cast(new_block->owningGraph());
    auto inputs = op->inputs();
    auto node_it = inputs.begin();
    auto scalar_it = op->scalar_args.begin();
    for (auto arg_type : op->cconv) {
      py::object obj;
      if (arg_type == 'c') {
        TORCH_CHECK(
            scalar_it != op->scalar_args.end(),
            "expected too many scalar args");
        obj = py::reinterpret_borrow<py::object>(
            py::handle((scalar_it++)->get()));
      } else if (arg_type == 'd') {
        TORCH_CHECK(node_it != inputs.end(), "expected too many inputs");
        obj = py::cast(envFn(*node_it++));
      } else {
        throw std::runtime_error("unexpected calling convention");
      }
      py_symbolic_args[input_nr++] = obj;
    }

    WithInsertPoint insert_point_guard(new_block);
    WithCurrentScope scope_guard(*new_block->owningGraph(), op->scope());
    // Call the symbolic function
    // Use a little trampoline function so we can give good error messages
    // upon argument mismatch
    py::object opset_version = onnx_symbolic.attr("_export_onnx_opset_version");
    onnx_registry.attr("register_op")(
        op->name(), pyobj.attr("symbolic"), "", opset_version);
    py::object raw_output = onnx.attr("_run_symbolic_method")(
        op->name(), pyobj.attr("symbolic"), py_symbolic_args);

    processSymbolicOutput(op->name(), op, raw_output);
  };

  auto k = old_node->kind();
  if (k.is_caffe2()) {
    // Pass on Caffe2 operator, since we already preprocess it
    cloneNode(old_node);
  } else if (k == prim::PythonOp) {
    callPySymbolicMethod(static_cast<ConcretePythonOp*>(old_node));
  } else {
    callPySymbolicFunction(old_node);
  }
}

} // namespace jit
} // namespace torch
