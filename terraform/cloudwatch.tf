resource "aws_cloudwatch_log_group" "training_jobs" {
  name              = "/aws/sagemaker/TrainingJobs"
  retention_in_days = 7

  tags = {
    Name = "SageMaker Training Jobs Logs"
    Note = "Individual log streams cannot be deleted via Terraform; see pre-destroy.sh"
  }

  lifecycle {
    prevent_destroy = false
  }
}

resource "aws_cloudwatch_log_group" "processing_jobs" {
  name              = "/aws/sagemaker/ProcessingJobs"
  retention_in_days = 7

  tags = {
    Name = "SageMaker Processing Jobs Logs"
    Note = "Individual log streams cannot be deleted via Terraform; see pre-destroy.sh"
  }

  lifecycle {
    prevent_destroy = false
  }
}
