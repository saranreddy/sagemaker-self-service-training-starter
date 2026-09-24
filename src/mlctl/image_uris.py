"""SageMaker container image URIs by region and framework."""

# sklearn/xgboost account IDs per region
# Source: https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html  # noqa: E501
SKLEARN_XGBOOST_ACCOUNTS = {
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-west-1": "746614075791",
    "us-west-2": "246618743249",
    "eu-west-1": "141502667606",
    "eu-central-1": "492215442770",
    "ap-southeast-1": "121021644041",
    "ap-southeast-2": "783357654285",
    "ap-northeast-1": "354813040037",
    "ap-northeast-2": "366743142698",
}

# PyTorch (training and inference) uses 763104351884 in all regions
PYTORCH_ACCOUNT = "763104351884"


def get_training_image_uri(framework: str, region: str) -> str:
    """Get training image URI for framework and region."""
    if framework == "pytorch":
        account = PYTORCH_ACCOUNT
        return (
            f"{account}.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-cpu-py310"
        )
    else:
        if region not in SKLEARN_XGBOOST_ACCOUNTS:
            raise ValueError(f"Unsupported region: {region}")
        account = SKLEARN_XGBOOST_ACCOUNTS[region]

        uris = {
            "sklearn": f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",  # noqa: E501
            "xgboost": f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1",
        }

        if framework not in uris:
            raise ValueError(f"Unsupported framework: {framework}")

        return uris[framework]


def get_inference_image_uri(framework: str, region: str) -> str:
    """Get inference image URI for framework and region (for model registration)."""
    if framework == "pytorch":
        return f"{PYTORCH_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.1.0-cpu-py310"
    else:
        return get_training_image_uri(framework, region)
