"""List all projects and their status."""
import sys
from collections import defaultdict

import boto3
import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command('list')
def list_projects():
    """List all projects: owner team, last run status, latest registered version."""
    session = boto3.session.Session()
    region = session.region_name or 'us-east-1'
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        pipelines_response = sagemaker_client.list_pipelines(
            MaxResults=100
        )
        
        pipelines = pipelines_response.get('PipelineSummaries', [])
        
        if not pipelines:
            console.print("[yellow]No pipelines found[/yellow]")
            sys.exit(0)
        
        table = Table(title="Projects")
        table.add_column("Project")
        table.add_column("Team")
        table.add_column("Last Run Status")
        table.add_column("Latest Model Version")
        
        for pipeline in pipelines:
            pipeline_name = pipeline['PipelineName']
            
            if not pipeline_name.endswith('-pipeline'):
                continue
            
            project_name = pipeline_name.replace('-pipeline', '')
            
            team = "unknown"
            last_status = "N/A"
            
            try:
                exec_response = sagemaker_client.list_pipeline_executions(
                    PipelineName=pipeline_name,
                    MaxResults=1,
                    SortBy='CreationTime',
                    SortOrder='Descending'
                )
                
                if exec_response.get('PipelineExecutionSummaries'):
                    last_status = exec_response['PipelineExecutionSummaries'][0]['PipelineExecutionStatus']
            except:
                pass
            
            model_version = "N/A"
            model_package_group = f"{project_name}-models"
            
            try:
                models_response = sagemaker_client.list_model_packages(
                    ModelPackageGroupName=model_package_group,
                    MaxResults=1,
                    SortBy='CreationTime',
                    SortOrder='Descending'
                )
                
                if models_response.get('ModelPackageSummaryList'):
                    model_version = str(models_response['ModelPackageSummaryList'][0].get('ModelPackageVersion', 'N/A'))
            except:
                pass
            
            table.add_row(project_name, team, last_status, model_version)
        
        console.print(table)
    
    except Exception as e:
        console.print(f"[red]Failed to list projects: {e}[/red]")
        sys.exit(1)
