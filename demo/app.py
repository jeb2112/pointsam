import numpy as np
# from point_sam import build_point_sam
import os
import numpy as np
import argparse
import gc
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request
from flask_cors import CORS
import hydra
from omegaconf import OmegaConf
from accelerate.utils import set_seed

import torch
from transformers import AutoModel, SamModel, AutoTokenizer
from safetensors.torch import load_model

from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from utils import load_ply,loadnifti


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default="/media/jbishop/WD4/brainmets/sam_models/psam")
parser.add_argument("--pointcloud", type=str, default="scene.ply")
parser.add_argument(
    "--config", type=str, default="large", help="path to config file"
)
parser.add_argument("--config_dir", type=str, default="../configs")
args, unknown_args = parser.parse_known_args()

# run in demo dir with app.app
# run in pointsam dir with what path? have to chdir instead
if True:
    os.chdir('/home/src/pointsam/demo')
os.environ['FLASK_APP'] = 'app.app'

# PCSAM variables
pc_xyz, pc_rgb = None, None
prompts, labels = [], []
prompt_mask = None
obj_path = None
output_dir = "results"
segment_mask = None
masks = []

# Flask Backend
app = Flask(__name__, static_folder="static")
CORS(
    app, origins=f"{args.host}:{args.port}", allow_headers="Access-Control-Allow-Origin"
)

# change "./pretrained/model.safetensors" to the path of the checkpoint
# sam = build_point_sam("./pretrained/model.safetensors").cuda()
# AutoModel from_pretrained has no keyword from_safetensor, but appears to load by default without it
# sam = AutoModel.from_pretrained('/media/jbishop/WD4/brainmets/sam_models/psam', from_safetensors=True)
if False:
    sam = AutoModel.from_pretrained('/media/jbishop/WD4/brainmets/sam_models/psam')
else:

    # ---------------------------------------------------------------------------- #
    # Load configuration
    # ---------------------------------------------------------------------------- #
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
        OmegaConf.resolve(cfg)
        # print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", 42)

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    set_seed(seed)
    sam: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    if False:
        sam.apply(replace_with_fused_layernorm)

    # ---------------------------------------------------------------------------- #
    # Load pre-trained model
    # ---------------------------------------------------------------------------- #
    load_model(sam, os.path.join(args.checkpoint,'model.safetensors'))
    sam.eval()
    sam.cuda()

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/static/<path:path>")
def static_server(path):
    return app.send_static_file(path)


@app.route("/mesh/<path:path>")
def mesh_server(path):
    # path = f"/home/yuchen/workspace/annotator_3d/src/static/models/{path}"
    # print(path)
    # path = "/home/yuchen/workspace/annotator_3d/src/static/models/Rhino/White Rhino.obj"
    path = f"models/{path}"
    print(path)
    return app.send_static_file(path)


@app.route("/sampled_pointcloud", methods=["POST"])
def sampled_pc():
    request_data = request.get_json()
    points = request_data["points"].values()
    points = np.array(list(points)).reshape(-1, 3)
    colors = request_data["colors"].values()
    colors = np.array(list(colors)).reshape(-1, 3)

    global pc_xyz, pc_rgb, data
    pc_xyz, pc_rgb = (
        torch.from_numpy(points).cuda().float(),
        torch.from_numpy(colors).cuda().float(),
    )
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)
    data = {"xyz": points, "rgb": colors, "mask": labels}

    response = "success"
    return jsonify({"response": response})


@app.route("/pointcloud/<path:path>")
def pointcloud_server(path):
    path = args.pointcloud
    global obj_path
    obj_path = path
    if 'ply' in path:
        points = load_ply(f"./static/models/{path}")
        xyz = points[:, :3]
        rgb = points[:, 3:6] / 255
    elif 'nii' in path:
        img_arr_t1,_ = loadnifti(os.path.split(path)[1],os.path.split(path)[0],type='float')
        img_arr_t1 = img_arr_t1[:,:,121:122]
        points = np.where(img_arr_t1)
        xyz = np.array(points).T
        rgb = img_arr_t1[points]
        rgb /= np.max(rgb)
        rgb = np.tile(rgb[:,np.newaxis],(1,3))
    elif 'testsmall' in path:
        arr = np.ones((64,64,1),dtype='float')
        points = np.where(arr)
        xyz = np.array(points).T
        rgb = np.ones((64,64,3),dtype='float')*0.1
        rgb[16:48,16:48]=[0.5,0.5,0]
        rgb[20:32,20:32] = [0,0.5,0.5]
        rgb = np.reshape(rgb,(4096,3))
    elif 'testlarge' in path:
        arr = np.ones((256,256,4),dtype='float')
        points = np.where(arr)
        xyz = np.array(points).T
        rgb = np.ones((256,256,4,3),dtype='float')*0.1
        rgb[64:192,64:192,:]=[0.5,0.5,0]
        rgb[96:112,96:112,:] = [0,0.5,0.5]
        rgb = np.reshape(rgb,(262144,3))
    elif 'test3d' in path:
        arr = np.ones((64,64,16),dtype='float')
        points = np.where(arr)
        xyz = np.array(points).T
        rgb = np.ones((64,64,16,3),dtype='float')*0.1
        rgb[16:48,16:48,:]=[0.5,0.5,0]
        rgb[20:32,20:32,:] = [0,0.5,0.5]
        rgb = np.reshape(rgb,(65536,3))
    elif 'testhuge' in path:
        arr = np.ones((1024,1024,1),dtype='float')
        points = np.where(arr)
        xyz = np.array(points).T
        rgb = np.ones((1024,1024,3),dtype='float')*0.1
        rgb[256:784,256:784]=[0.5,0.5,0]
        rgb[384:440,384:440] = [0,0.5,0.5]
        rgb = np.reshape(rgb,(1048576,3))
    # print(rgb.max())
    # indices = np.random.choice(xyz.shape[0], 30000, replace=False)
    # xyz = xyz[indices]
    # rgb = rgb[indices]

    # normalize
    shift = xyz.mean(0)
    scale = np.linalg.norm(xyz - shift, axis=-1).max()
    if True:
        xyz = (xyz - shift) / scale
    else:
        xyz = xyz - shift

    # set pcsam variables
    global pc_xyz, pc_rgb, data
    pc_xyz, pc_rgb = (
        torch.from_numpy(xyz).cuda().float(),
        torch.from_numpy(rgb).cuda().float(),
    )
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)
    # data = {"xyz": points, "rgb": colors, "mask": labels}

    # flatten
    xyz = xyz.flatten()
    rgb = rgb.flatten()

    return jsonify({"xyz": xyz.tolist(), "rgb": rgb.tolist()})


@app.route("/clear", methods=["POST"])
def clear():
    global prompts, labels, prompt_mask, segment_mask
    prompts, labels = [], []
    prompt_mask = None
    segment_mask = None
    return jsonify({"status": "cleared"})


@app.route("/next", methods=["POST"])
def next():
    global prompts, labels, segment_mask, masks, prompt_mask
    masks.append(segment_mask.cpu().numpy())
    prompts, labels = [], []
    prompt_mask = None
    return jsonify({"status": "cleared"})


@app.route("/save", methods=["POST"])
def save():
    os.makedirs(output_dir, exist_ok=True)
    global pc_xyz, pc_rgb, segment_mask, obj_path, masks
    xyz = pc_xyz[0].cpu().numpy()
    rgb = pc_rgb[0].cpu().numpy()
    masks = np.stack(masks)
    obj_path = obj_path.split(".")[0]
    np.save(f"{output_dir}/{obj_path}.npy", {"xyz": xyz, "rgb": rgb, "mask": masks})
    global prompts, labels, prompt_mask
    prompts, labels = [], []
    prompt_mask = None
    segment_mask = None
    return jsonify({"status": "saved"})


@app.route("/segment", methods=["POST"])
def segment():
    request_data = request.get_json()
    print(request_data['prompt_point'],request_data['prompt_label'])
    prompt_point = request_data["prompt_point"]
    prompt_label = request_data["prompt_label"]

    # append prompt
    global prompts, labels, prompt_mask
    prompts.append(prompt_point)
    labels.append(prompt_label)

    prompt_points = torch.from_numpy(np.array(prompts)).cuda().float()[None, ...]
    prompt_labels = torch.from_numpy(np.array(labels)).cuda()[None, ...]

    data = {
        "points": pc_xyz,
        "rgb": pc_rgb,
        "prompt_points": prompt_points,
        "prompt_labels": prompt_labels,
        "prompt_mask": prompt_mask,
    }
    if True:
        with torch.no_grad():
            # sam.set_pointcloud(pc_xyz, pc_rgb)
            # mask, scores, logits = sam.predict_masks(
            #     prompt_points, prompt_labels, prompt_mask, prompt_mask is None
            # )
            mask, scores = sam.predict_masks(
                pc_xyz,pc_rgb,prompt_points, prompt_labels, multimask_output=True
            )
    else:
        data = {"coords": pc_xyz.cpu().numpy(), "features": pc_rgb.cpu().numpy(), "gt_masks": labels}
        outputs = sam(**data, is_eval=True)

    if False:
        prompt_mask = logits[0][torch.argmax(scores[0])][None, ...]
    global segment_mask
    if False:
        segment_mask = return_mask = mask[0][torch.argmax(scores[0])] > 0
    else:
        segment_mask = return_mask = mask[0][1] > 0
        # segment_mask = return_mask = mask[0][0] > 0
    return jsonify({"seg": return_mask.cpu().numpy().tolist()})


if __name__ == "__main__":
    # something about a hot reloader when in debug mode, which double-allocates tensors on the gpu
    # force it not to use the reloader if short of gpu memory
    app.run(host=f"{args.host}", port=f"{args.port}", debug=True, use_reloader=False)
