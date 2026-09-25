resource "aws_s3_bucket" "artifact_bucket" {
  bucket        = var.artifact_bucket_name != "" ? var.artifact_bucket_name : "${var.project_name}-artifacts-${local.account_id}"
  force_destroy = true

  tags = {
    Name = "${var.project_name}-artifacts"
    Note = "force_destroy is enabled for demo purposes - disable for production"
  }
}

resource "aws_s3_bucket_public_access_block" "artifact_bucket" {
  bucket = aws_s3_bucket.artifact_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifact_bucket" {
  bucket = aws_s3_bucket.artifact_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "artifact_bucket" {
  bucket = aws_s3_bucket.artifact_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}
