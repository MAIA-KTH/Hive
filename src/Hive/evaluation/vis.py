import json
import os

import visdom


# SAVING


def create_log_at(file_path, current_env, new_env=None):
    new_env = current_env if new_env is None else new_env
    vis = visdom.Visdom(env=current_env)

    data = json.loads(vis.get_window_data())
    if len(data) == 0:
        print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
        return

    file = open(file_path, "w+")
    for datapoint in data.values():
        output = {"win": datapoint["id"], "eid": new_env, "opts": {}}

        if datapoint["type"] != "plot":
            output["data"] = [{"content": datapoint["content"], "type": datapoint["type"]}]
            if datapoint["height"] is not None:
                output["opts"]["height"] = datapoint["height"]
            if datapoint["width"] is not None:
                output["opts"]["width"] = datapoint["width"]
        else:
            output["data"] = datapoint["content"]["data"]
            output["layout"] = datapoint["content"]["layout"]

        to_write = json.dumps(["events", output])
        file.write(to_write + "\n")
    file.close()


def create_log(current_env, new_env=None):
    new_env = current_env if new_env is None else new_env
    dir_path = os.getcwd()
    if not os.path.exists(dir_path + "/log"):
        os.makedirs(dir_path + "/log")
    file_path = dir_path + "/log/" + new_env + ".log"
    create_log_at(file_path, current_env, new_env)


# LOADING


def load_log_at(path):
    visdom.Visdom().replay_log(path)


def load_log(env):
    dir_path = os.getcwd()
    load_log_at(dir_path + "/log/" + env + ".log")


def load_all_log():
    dir_path = os.getcwd() + "/log/"
    logs = os.listdir(dir_path)
    vis = visdom.Visdom()
    for log in logs:
        vis.replay_log(dir_path + log)
