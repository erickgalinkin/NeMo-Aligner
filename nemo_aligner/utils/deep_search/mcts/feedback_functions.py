import os
import re

import pandas as pd
import numpy as np
from datasets import load_dataset
from nemo_skills.code_execution.math_grader import extract_answer
from nemo_skills.code_execution.sandbox import LocalSandbox

from nemo_aligner.utils.deep_search.mcts.reward_functions import get_reward, get_helpfulness_reward, get_harmfulness_reward
from nemo_aligner.utils.ngc_llm_inference import connect_llm_service, run_inference


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Feedback(object):
    def __init__(self):
        pass

    def score(self, response, context_id):
        """
        score the response
        """
        raise NotImplementedError


class DummyScore(Feedback):
    def score(self, response, data_id):
        return 0.0


class GSK8KFeedbackDataset(Feedback):
    def __init__(self, ds):
        self.ds = ds
        # local_rank = os.getenv("local_rank", "0")
        host = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "localhost")
        port = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "1034")
        self.sandbox = LocalSandbox(host=host, port=port)

    def score(self, response, data_id):
        """
        score the response
        """
        assert self.ds[data_id]["data_id"] == data_id
        response = response.lower()
        answer = self.ds[data_id]["expected_answer"]
        # this needs to be on a seperate server for anything
        # complicated but for GSM8K this is fine
        response = extract_answer(response)
        try:
            score = float(self.sandbox.is_output_correct(response, answer))
        except Exception as e:
            print("############ Inference failed ############")
            print(answer, response)
            print(e)
            score = 0.0
        finally:
            return score


class SteerLMFeedback(Feedback):
    def __init__(self):
        # local_rank = os.getenv("local_rank", "0")
        self.host = os.getenv("REWARD_SERVER_HOST", "localhost")
        self.port = os.getenv("REWARD_SERVER_PORT", "1234")

    def score(self, response, data_id):
        """
        score the response
        """
        # remove the trailing extra_id_1
        if response.endswith("<extra_id_1>"):
            response = response[: -len("<extra_id_1>")]
        # get the expected answer, e.g. 'quality:4,toxicity:0,humor:0,creativity:0,helpfulness:4,correctness:4,coherence:4,complexity:4,verbosity:2'
        attribute_str = response.split("<extra_id_2>")[-1].split("\n")[0]
        # extract the numbers
        attributes = attribute_str.split(",")
        numbers = [int(attr.split(":")[-1]) for attr in attributes]
        # remove the <extra_id_2> line
        response = "\n".join([i for i in response.split("\n") if not i.startswith("<extra_id_2>")])
        response = response + "<extra_id_2>"
        try:
            evaluate = get_reward([response], False, self.host, self.port)[0]

            # compute the distance between the two vectors
            distance = sum([int(bool(a - b)) for a, b in zip(numbers, evaluate)])

            # normalize the distance to be between 0 and 1
            distance = distance / (len(numbers))

            score = 1 - distance
        except Exception as e:
            print("############ Inference failed ############")
            print(e)
            score = 0.0
        finally:
            return score


class GSK8KFeedback(Feedback):
    def score(self, response, answer):
        """
        score the response
        """
        response = response.lower()
        answer = answer.lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0


class GSK8KFeedback(Feedback):
    def __init__(self):
        ...

    def score(self, response, answer):
        """
        score the response
        """
        response = response.lower()
        answer = answer.lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0


class GSK8KFeedbackHF(Feedback):
    def __init__(self, split):
        super().__init__()
        self.ds = load_dataset("gsm8k", "main")
        self.split = split

    def score(self, response, data_id):
        """
        score the response
        """
        response = response.lower()
        answer = self.ds[self.split][data_id]["answer"].lower().split("####")[1].strip().replace(",", "")
        # predicted answer matches the answer pattern
        numbers = re.findall(r"\{{([\d,]+)\}}", response)
        # Extract the last number
        last_number = numbers[-1] if numbers else None
        if last_number is None:
            return 0.0
        if last_number == answer:
            return 1.0
        else:
            return 0.0


class HelpfulnessFeedback(Feedback):
    def __init__(self):
        # local_rank = os.getenv("local_rank", "0")
        self.host = os.getenv("HELPFUL_REWARD_SERVER_HOST", "localhost")
        self.port = os.getenv("HELPFUL_REWARD_SERVER_PORT", "1424")

    def score(self, response, data_id=None):
        """
        score the response
        """
        # remove the trailing extra_id_1
        if response.endswith("<extra_id_1>"):
            response = response[: -len("<extra_id_1>")]
        # remove the <extra_id_2> line
        response = "\n".join([i for i in response.split("\n") if not i.startswith("<extra_id_2>")])
        try:
            evaluate = get_helpfulness_reward([response], False, self.host, self.port)[0]
            score = sigmoid(evaluate[0])
        except Exception as e:
            print("############ Inference failed ############")
            print(e)
            score = 0.0
        finally:
            return score


class HarmfulnessFeedback(Feedback):
    def __init__(self):
        ngc_api_key = os.environ.get("NGC_API_KEY")
        ngc_org_id = os.environ.get("NGC_ORG_ID")
        customization_id = os.environ.get("NGC_CUSTOMIZATION_ID")
        self.conn = connect_llm_service(ngc_api_key=ngc_api_key, org_id=ngc_org_id)
        self.run_inference = run_inference
        self.customization_id = customization_id

    def score(self, response, data_id=None):
        """
        score the response
        """
        try:
            score = run_inference(conn_obj=self.conn, prompt=response, customization_id=self.customization_id)
        except Exception as e:
            print("############ Inference failed ############")
            print(e)
            score = 0.0
        finally:
            return score


class JailbreakFeedback(Feedback):
    def __init__(self):
        super().__init__()
        self.helpful = HelpfulnessFeedback()
        self.harmful = HarmfulnessFeedback()

    def score(self, response, data_id=None):
        """
        score the response
        """
        helpful_score = self.helpful.score(response)
        harmful_score = self.harmful.score(response)
        overall_score = helpful_score - harmful_score
        return overall_score
