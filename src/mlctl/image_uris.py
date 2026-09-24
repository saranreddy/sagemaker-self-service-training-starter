"""SageMaker Deep Learning Container image URIs by region and framework."""

# AWS DLC account IDs per region
# Source: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
DLC_ACCOUNTS = {
    "us-east-1": "683313688378",
    "us-east-2": "257758044811",
    "us-west-1": "746614075791",
    "us-west-2": "246618743249",
    "eu-west-1": "685385470294",
    "eu-central-1": "492215442770",
    "ap-southeast-1": "192199979996",
    "ap-southeast-2": "666831318237",
    "ap-northeast-1": "354813040037",
    "ap-northeast-2": "366743142698",
}

# PyTorch inference uses different account
PYTORCH_INFERENCE_ACCOUNTS = {
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
    "eu-west-1": "763104351884",
    "eu-central-1": "763104351884",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ap-northeast-1": "763104351884",
    "ap-northeast-2": "763104351884",
}


def get_training_image_uri(framework: str, region: str) -> str:
    """Get training image URI for framework and region."""
    if region not in DLC_ACCOUNTS:
        raise ValueError(f"Unsupported region: {region}")

    account = DLC_ACCOUNTS[region]

    uris = {
        "sklearn": f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        "xgboost": f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1",
        "pytorch": f"{account}.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.1.0-cpu-py310",
    }

    if framework not in uris:
        raise ValueError(f"Unsupported framework: {framework}")

    return uris[framework]


def get_inference_image_uri(framework: str, region: str) -> str:
    """Get inference image URI for framework and region (for model registration)."""
    if region not in DLC_ACCOUNTS:
        raise ValueError(f"Unsupported region: {region}")

    if framework == "pytorch":
        account = PYTORCH_INFERENCE_ACCOUNTS.get(region, "763104351884")
        return f"{account}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.1.0-cpu-py310"
    else:
        return get_training_image_uri(framework, region)
