"""Architecture diagram for saranreddy/sagemaker-self-service-training-starter.

Verified against main @ a685a7e. Every node maps to terraform/*.tf or src/mlctl/
(commands/submit.py, packaging.py, pipeline.py).

Render:  pip install diagrams   (also needs Graphviz: apt install graphviz / brew install graphviz)
         python docs/architecture.py   ->  docs/architecture.png (written next to this script)
"""
import os

from diagrams import Cluster, Diagram, Edge, getdiagram
from diagrams import Node
from diagrams.aws.compute import EC2ContainerRegistryImage
from diagrams.aws.general import User
from diagrams.aws.management import CloudwatchLogs
from diagrams.aws.ml import Sagemaker, SagemakerModel, SagemakerTrainingJob
from diagrams.aws.security import IAMRole
from diagrams.aws.storage import SimpleStorageServiceS3Bucket, SimpleStorageServiceS3BucketWithObjects
from diagrams.onprem.iac import Terraform
from diagrams.programming.flowchart import Decision
from diagrams.programming.language import Python

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "architecture")  # -> architecture.png next to this script

FONT = "DejaVu Sans"
GRAPH = {
    "fontname": FONT, "fontsize": "34", "labelloc": "t", "pad": "0.4",
    "nodesep": "0.4", "ranksep": "1.0", "splines": "spline", "newrank": "true",
    "compound": "true",
}
NODE = {"fontname": FONT, "fontsize": "21", "imagepos": "tc"}
EDGE = {"fontname": FONT, "fontsize": "19", "color": "#555555",
        # enter/leave icons at mid-height so arrowheads never land on label text
        "tailport": "e", "headport": "w"}

# diagrams.Edge hard-codes a 13pt label font on every edge; raise it so edge labels stay
# readable when the PNG is scaled down to README width.
Edge._default_edge_attrs = {"fontcolor": "#2D3436", "fontname": FONT, "fontsize": "19"}


def box(bg, pen, style="rounded"):
    return {"bgcolor": bg, "pencolor": pen, "fontname": FONT, "fontsize": "21",
            "style": style, "labeljust": "l", "margin": "24"}


TF_BOX = box("#fff4e0", "#e66100")                  # deployed by Terraform
SUB_BOX = box("#fffaf2", "#e66100")                 # sub-group inside a Terraform box
RUN_BOX = box("#e8f1fb", "#1a5fb4")                 # created by scripts / CLI
EXEC_BOX = box("#f3eefa", "#613583")                # per execution / runtime
MANAGED_BOX = box("#f6f5f4", "#9a9996", "dashed")   # not created by this repo
ACCOUNT_BOX = box("#ffffff", "#232f3e")

FLOW = dict(color="#1a5fb4", fontcolor="#1a5fb4", penwidth="2.2")
IO = dict(color="#26a269", fontcolor="#1e7d4f", penwidth="1.8")
IAM = dict(color="#c01c28", fontcolor="#c01c28", style="dashed", penwidth="1.6", constraint="false")
AUX = dict(color="#8a8a8a", fontcolor="#5e5c64", style="dotted", penwidth="1.8")
SETUP = dict(color="#e66100", fontcolor="#c64600", style="dashed", penwidth="1.8")
MANUAL = dict(color="#26a269", fontcolor="#1e7d4f", style="dashed", penwidth="2.2")
FAIL = dict(color="#c01c28", fontcolor="#c01c28", penwidth="2.2")
OPT = dict(color="#b5835a", fontcolor="#8f5f3a", style="dashed", penwidth="1.8")
HIDDEN = dict(style="invis")
DOWN = dict(tailport="s", headport="n")
UP = dict(tailport="n", headport="s")


def same_rank(*nodes):
    getdiagram().dot.body.append("{rank=same; " + " ".join(f'"{n._id}";' for n in nodes) + "}")


# Top-to-bottom layout keeps the pipeline on one row, so the PNG stays narrow enough
# to read at README width.
GRAPH["ranksep"] = "0.9"
GRAPH["pad"] = "0.8"
GRAPH["nodesep"] = "1.2"
GRAPH["forcelabels"] = "true"
EDGE_TB = {k: v for k, v in EDGE.items() if k not in ("tailport", "headport")}
ROW = dict(tailport="e", headport="w")     # flat edge inside a row, left -> right
# plain red box for the pipeline Fail step (no AWS icon exists for it)
FAIL_STEP = dict(shape="box", style="rounded,filled", fillcolor="#fbe3e4", color="#c01c28",
                 fixedsize="false", width="2.6", height="1.1", labelloc="c", margin="0.2")

with Diagram(
    "SageMaker self-service training starter (mlctl)",
    filename=OUT, outformat="png", show=False, direction="TB",
    graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE_TB,
):
    mlops = User("MLOps engineer")
    ds = User("Data scientist")
    tf = Terraform("terraform apply\n(make apply)")
    cli = Python("mlctl submit\n(Python + boto3)")

    with Cluster("AWS account  (region from AWS config, else us-east-1)", graph_attr=ACCOUNT_BOX):
        with Cluster("Deployed once by Terraform", graph_attr=TF_BOX):
            role = IAMRole("Execution role\n<project>-\nexecution-role\n+ inline policy")
            bucket = SimpleStorageServiceS3Bucket("Artifact bucket\n<project>-\nartifacts-<acct>\n(versioned)")

        with Cluster("Created / updated by mlctl submit", graph_attr=RUN_BOX):
            pipeline = Sagemaker("SageMaker\nPipeline\n<name>-\npipeline")
            code = SimpleStorageServiceS3BucketWithObjects("Code package\n(artifact bucket)\ncode/<name>/\n<git>-<hash>/")
            mpg = SagemakerModel("Model package\ngroup\n<name>-models\n(if missing)")

        data = SimpleStorageServiceS3BucketWithObjects("Training data\n(S3 URIs in\nml.yaml, staged\nby the user)")

        with Cluster("Per pipeline execution", graph_attr=EXEC_BOX) as run:
            train = SagemakerTrainingJob("TrainModel\ntraining job")
            evaluate = Sagemaker("EvaluateModel\nprocessing job")
            gate = Decision("QualityGate\nCheck")
            version = SagemakerModel("RegisterModel\nversion, Pending\nManualApproval")
            failed = Node("QualityGateFailed\n(Fail step): stops,\nnothing registered", **FAIL_STEP)

        with Cluster("AWS-managed", graph_attr=MANAGED_BOX):
            logs = CloudwatchLogs("CloudWatch Logs\n/aws/sagemaker/\nTrainingJobs,\nProcessingJobs")
            ecr = EC2ContainerRegistryImage("Framework\nimages: sklearn/\nxgboost/pytorch")

    approver = User("MLOps engineer\n(approver)")

    # row 1: people and tools
    mlops >> Edge(xlabel="deploys", **SETUP) >> tf
    ds >> Edge(xlabel="runs", **FLOW) >> cli
    tf >> Edge(xlabel="org-config.yaml\n(role ARN, bucket)", **ROW, **SETUP) >> cli
    same_rank(tf, cli)

    # row 2: Terraform resources + what mlctl submit creates
    tf >> Edge(**SETUP) >> role
    tf >> Edge(**SETUP) >> bucket
    cli >> Edge(label="1. upload code", **FLOW) >> code
    cli >> Edge(label="2. ensure group", **FLOW) >> mpg
    cli >> Edge(label="3. create/update\n+ start", **FLOW) >> pipeline
    role >> Edge(label="passed to SageMaker", **IAM) >> pipeline
    same_rank(role, bucket, code, pipeline, mpg)

    # row 3: pipeline steps
    pipeline >> Edge(label="runs", **FLOW) >> train
    code >> Edge(label="code", **IO) >> train
    code >> Edge(**IO) >> evaluate
    data >> Edge(xlabel="train / val", **ROW, **IO) >> train
    data >> Edge(label="test (else val)", constraint="false", **IO) >> evaluate
    train >> Edge(xlabel="model\n.tar.gz", **ROW, **FLOW) >> evaluate
    evaluate >> Edge(xlabel="metrics\n.json", **ROW, **FLOW) >> gate
    gate >> Edge(xlabel="pass", color="#26a269", fontcolor="#1e7d4f", penwidth="2.2", **ROW) >> version
    same_rank(data, train, evaluate, gate, version)
    bucket << Edge(label="step outputs:\npipelines/<execId>/", lhead=run.name, **IO) << train
    mpg << Edge(label="version of", style="dashed", tailport="s", headport="n", **FLOW) << version

    # row 4: failure branch, approval, AWS-managed support
    gate >> Edge(label="fail", **FAIL) >> failed
    version << Edge(label="approve\n(console / CLI)", **MANUAL) << approver
    data >> Edge(**HIDDEN) >> logs
    train >> Edge(**HIDDEN) >> ecr
    ecr >> Edge(xlabel="images", lhead=run.name, constraint="false", **AUX) >> train
    train >> Edge(xlabel="job logs", ltail=run.name, constraint="false", **AUX) >> logs
    same_rank(ecr, logs, failed, approver)
