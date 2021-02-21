import cimodel.lib.miniutils as miniutils

class MacOsJob:
    def __init__(self, os_version, is_build=False, is_test=False, extra_props={}):
        self.os_version = os_version
        self.is_build = is_build
        self.is_test = is_test
        self.extra_props = extra_props

    def gen_tree(self):
        non_phase_parts = ["pytorch", "macos", self.os_version, "py3"]

        extra_name_list = [name for name, exist in self.extra_props.items() if exist]
        full_job_name_list = non_phase_parts + extra_name_list + [
            'build' if self.is_build else None,
            'test' if self.is_test else None,
        ]

        full_job_name = "_".join(list(filter(None, full_job_name_list)))

        # phase_name = "test" if self.is_test else "build"
        # full_job_name = "_".join(non_phase_parts + [extra_name])

        test_build_dependency = "_".join(non_phase_parts + ["build"])
        extra_dependencies = [test_build_dependency] if self.is_test else []
        job_dependencies = extra_dependencies

        # Yes we name the job after itself, it needs a non-empty value in here
        # for the YAML output to work.
        props_dict = {"requires": job_dependencies, "name": full_job_name}

        # if self.extra_props:
        #     props_dict.update(self.extra_props)

        return [{full_job_name: props_dict}]


WORKFLOW_DATA = [
    MacOsJob("10_15", is_build=True),
    MacOsJob("10_13", is_build=True),
    MacOsJob(
        "10_13",
        is_build=False,
        is_test=True,
    ),
    MacOsJob(
        "10_13",
        is_build=True,
        is_test=True,
        extra_props={
            "lite_interpreter": True,
        },
    )
]


def get_workflow_jobs():
    return [item.gen_tree() for item in WORKFLOW_DATA]
