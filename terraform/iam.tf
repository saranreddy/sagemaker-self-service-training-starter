data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    effect = "Allow"

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }

    actions = ["sts:AssumeRole"]

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:sagemaker:${local.region}:${local.account_id}:*"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${var.project_name}-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = {
    Name = "${var.project_name}-execution-role"
  }
}

data "aws_iam_policy_document" "sagemaker_execution_policy" {
  statement {
    sid    = "S3Access"
    effect = "Allow"

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket"
    ]

    resources = [
      aws_s3_bucket.artifact_bucket.arn,
      "${aws_s3_bucket.artifact_bucket.arn}/*"
    ]
  }

  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"

    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams"
    ]

    resources = [
      "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*"
    ]
  }

  statement {
    sid    = "ECRPull"
    effect = "Allow"

    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetAuthorizationToken"
    ]

    resources = ["*"]
  }

  statement {
    sid    = "SageMakerTraining"
    effect = "Allow"

    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:AddTags"
    ]

    resources = [
      "arn:aws:sagemaker:${local.region}:${local.account_id}:training-job/*",
      "arn:aws:sagemaker:${local.region}:${local.account_id}:processing-job/*"
    ]
  }

  statement {
    sid    = "SageMakerPipelines"
    effect = "Allow"

    actions = [
      "sagemaker:DescribePipeline",
      "sagemaker:ListPipelineExecutions",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps"
    ]

    resources = [
      "arn:aws:sagemaker:${local.region}:${local.account_id}:pipeline/*"
    ]
  }

  statement {
    sid    = "SageMakerModelRegistry"
    effect = "Allow"

    actions = [
      "sagemaker:CreateModelPackageGroup",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:CreateModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:ListModelPackages",
      "sagemaker:AddTags"
    ]

    resources = [
      "arn:aws:sagemaker:${local.region}:${local.account_id}:model-package-group/*",
      "arn:aws:sagemaker:${local.region}:${local.account_id}:model-package/*"
    ]
  }

  statement {
    sid    = "PassRole"
    effect = "Allow"

    actions = [
      "iam:PassRole"
    ]

    resources = [
      aws_iam_role.sagemaker_execution_role.arn
    ]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  name   = "${var.project_name}-execution-policy"
  role   = aws_iam_role.sagemaker_execution_role.id
  policy = data.aws_iam_policy_document.sagemaker_execution_policy.json
}
