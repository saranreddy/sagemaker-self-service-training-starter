variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name (used for naming resources)"
  type        = string
  default     = "sagemaker-self-service-training"
}

variable "artifact_bucket_name" {
  description = "Name for the artifact S3 bucket (leave empty for auto-generated name)"
  type        = string
  default     = ""
}
