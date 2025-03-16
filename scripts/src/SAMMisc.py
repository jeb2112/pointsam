# miscellaneous classes

class PromptType:  
    CONTROL_POINTS = "pts"  
    BOUNDING_BOX = "bbox"
    
class TrainHistory():
    def __init__(self, history=None):
        self._history = history or []

    def start_session(
        self, 
        dataset_name: str,
        optimizer_name: str,
        loss_func: str,
        learning_rate: float,
        weight_decay: float,
        batch_size: int,
        prompt_args: dict,
    ):
        session = {
            "epochs": 0,
            "dataset": dataset_name, 
            "optimizer": optimizer_name,
            "loss_func": loss_func,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "prompt_args": prompt_args,
        }
        self._history.append(session)

    def update_session(self, epochs_elapsed: int):
        prev_elapsed = self._history[-1]["epochs"]
        if epochs_elapsed < prev_elapsed:
            print(f"Warning: Updated session to lower epoch value! previous={prev_elapsed}, requested={epochs_elapsed}")
        self._history[-1]["epochs"] = epochs_elapsed

    def get_history_list(self):
        return self._history

class LossFunctionsConfig():
    def __init__(self, funcs, track_iou=False):
        self.names = [item[0] for item in funcs]
        self.funcs = [item[1] for item in funcs]
        self.track_iou = track_iou
