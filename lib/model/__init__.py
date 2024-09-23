from typing import Any
from .classification import buildClassificationModel
from .segmentation import buildSegmentationModel

SUPPORTED_TASKS = ["classification", "segmentation"]

def getModel(opt, **kwargs: Any):
    Model = None

    # Debug statement: Print the task being initialized
    print(f"Initializing model for task: {opt.task}")

    if opt.task == "classification":
        # Debug statement: Indicating classification model initialization
        print("Building classification model...")
        Model = buildClassificationModel(opt, **kwargs)
    elif opt.task == "segmentation":
        # Debug statement: Indicating segmentation model initialization
        print("Building segmentation model...")
        Model = buildSegmentationModel(opt, **kwargs)
    else:
        # Raise an error if an unsupported task is provided
        TaskStr = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                   '\nSupported tasks are:'.format(opt.task)
        for i, Name in enumerate(SUPPORTED_TASKS):
            TaskStr += "\n\t {}: {}".format(i, Name)
        raise ValueError(TaskStr)

    # Check if the model was successfully initialized
    if Model is None:
        raise ValueError(f"Failed to initialize the model for task: {opt.task}")

    # Debug statement: Print success message for model initialization
    print(f"Model {opt.model_name} initialized successfully for task: {opt.task}")

    return Model
