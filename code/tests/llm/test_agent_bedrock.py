import io
import json
import os
import unittest
from unittest.mock import patch, MagicMock

from botocore.exceptions import ClientError
import sys
src_llm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'llm'))
if src_llm_path not in sys.path:
    sys.path.insert(0, src_llm_path)
from agent_bedrock import LLM

class TestLLM(unittest.TestCase):
    def setUp(self):
        os.environ["aws_access_key_id"] = "fake_access_key"
        os.environ["aws_secret_access_key"] = "fake_secret_key"
        os.environ["aws_session_token"] = "fake_session_token"

    def tearDown(self):
        os.environ.pop("aws_access_key_id", None)
        os.environ.pop("aws_secret_access_key", None)
        os.environ.pop("aws_session_token", None)

    @patch("agent_bedrock.boto3.Session")
    def test_create_client_success(self, mock_session):
        fake_client = MagicMock()
        instance = mock_session.return_value
        instance.client.return_value = fake_client

        llm = LLM(region_name="us-test-1")
        client = llm.create_client()

        mock_session.assert_called_with(
            aws_access_key_id="fake_access_key",
            aws_secret_access_key="fake_secret_key",
            aws_session_token="fake_session_token"
        )
        instance.client.assert_called_with("bedrock-runtime", region_name="us-test-1")
        self.assertEqual(client, fake_client)

    @patch("agent_bedrock.load_dotenv")
    def test_create_client_missing_credentials(self, mock_load_dotenv):
        os.environ.pop("aws_access_key_id", None)
        os.environ.pop("aws_secret_access_key", None)
        llm = LLM()
        with self.assertRaises(ValueError) as context:
            llm.create_client()
        self.assertIn("AWS credentials are not properly set", str(context.exception))

    def _configure_fake_client(self, generation_text):
        fake_client = MagicMock()
        fake_response = {
            "body": io.StringIO(json.dumps({"generation": generation_text}))
        }
        fake_client.invoke_model.return_value = fake_response
        return fake_client

    @patch("agent_bedrock.boto3.Session")
    def test_get_response_success(self, mock_session):
        fake_client = MagicMock()
        generation_text = "Test response from model."
        fake_response = {
            "body": io.StringIO(json.dumps({"generation": generation_text}))
        }
        fake_client.invoke_model.return_value = fake_response

        instance = mock_session.return_value
        instance.client.return_value = fake_client

        llm = LLM()

        llm.create_client()

        model_id = "model.instruct.test"
        body_content = {
            "prompt": "What is the dosage?",
            "temperature": 0.5,
            "max_gen_len": 512,
            "top_p": 0.9
        }
        response = llm.get_response(model_id, body_content)
        self.assertEqual(response, generation_text.strip())

        args, kwargs = fake_client.invoke_model.call_args
        invoked_body = json.loads(kwargs["body"])
        self.assertIn("<|begin_of_text|>", invoked_body["prompt"])

    @patch("agent_bedrock.boto3.Session")
    def test_get_response_client_error(self, mock_session):
        fake_client = MagicMock()
        error_response = {"Error": {"Message": "Simulated error"}}
        fake_client.invoke_model.side_effect = ClientError(error_response, "invoke_model")

        instance = mock_session.return_value
        instance.client.return_value = fake_client

        llm = LLM()
        llm.create_client()

        model_id = "model.test"
        body_content = {"prompt": "Test prompt", "temperature": 0.5, "max_gen_len": 512, "top_p": 0.9}
        response = llm.get_response(model_id, body_content)
        self.assertIsNone(response)

    @patch("agent_bedrock.boto3.Session")
    def test_get_response_unexpected_format(self, mock_session):
        fake_client = MagicMock()
        fake_response = {
            "body": io.StringIO(json.dumps({"unexpected": "data"}))
        }
        fake_client.invoke_model.return_value = fake_response

        instance = mock_session.return_value
        instance.client.return_value = fake_client

        llm = LLM()
        llm.create_client()

        model_id = "model.test"
        body_content = {"prompt": "Test prompt", "temperature": 0.5, "max_gen_len": 512, "top_p": 0.9}
        response = llm.get_response(model_id, body_content)
        self.assertIsNone(response)

    @patch("agent_bedrock.boto3.Session")
    def test_get_response_json_decode_error(self, mock_session):
        fake_client = MagicMock()
        fake_response = {
            "body": io.StringIO("Not a JSON string")
        }
        fake_client.invoke_model.return_value = fake_response

        instance = mock_session.return_value
        instance.client.return_value = fake_client
        llm = LLM()
        llm.create_client()
        model_id = "model.test"
        body_content = {"prompt": "Test prompt", "temperature": 0.5, "max_gen_len": 512, "top_p": 0.9}
        response = llm.get_response(model_id, body_content)
        self.assertIsNone(response)


if __name__ == '__main__':
    unittest.main()