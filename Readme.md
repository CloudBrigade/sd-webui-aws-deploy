## Deployment Scripts for Stable Diffusion Web UI on AWS

### Overview

The repository provides the scripts necessary to deploy Stable Diffusion WebUI from Automatic 1111. This includes the following components :

* CloudFormation Templates for various EC2 instance types
* Setup script to deploy all resources at provisioning time
* Systemd and Logrotate scripts
* config.json file with defaults for API access and CPU based inferencing for non-GPU accelerators

Currently supported AWS EC2 instance types are :

* G4DN - POWERED BY NVIDIA T4 GPUS
* DL1 - POWERED BY INTEL/HABANA GAUDI2

Note: Please check instance pricing as some instance types are expensive (dl1.24xl is $13/hr)

This repository is intended to provide fully automatic deployment of resources without logging into the Linux console. Deployment times may vary, but 15-20 minutes is typical.

### Considerations

Ubuntu22.04 is used by default. For special instance types such as the DL1, a marketplace image is required to provide CPU based accelerator libraries to the OS. You may need to presubscribe to these images in your AWS account.

Certain sensitive parameters are required for setup, such as the preferred EC2 KeyPair, and the SD-WebUI API credentials. AWS Systems Manager Parameter Store is used to store these items securely. Secrets Manager could also be used, but at additional cost.

### Getting Started

#### Prerequisites

You may need to login to the AWS Console to complete the following :

* Create and IAM user with programmatic access for AWS CLI
* Create a KeyPair "ec2-keypair-sd-webui" name in Parameter Store
https://console.aws.amazon.com/systems-manager/parameters/
* Subscribe to select EC2 images from the Marketplace (i.e. Habana) https://aws.amazon.com/marketplace/pp/prodview-5jwcnmim6brvc?sr=0-2&ref_=beagle&applicationId=AWS-Marketplace-Console

On your local workstation :

* Install the latest AWS-CLI from AWS https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
* Configure your programmatic access for AWS-CLI including your preferred region https://docs.aws.amazon.com/cli/latest/userguide/getting-started-prereqs.html

#### Review

Edit files in the repository as appropriate, such as the CloudFront templates, config.json, and setup.sh

Deploy the CloudFornation stack with the desired template and stack name :

aws cloudformation create-stack --stack-name sd-webui-habana-01 --template-body file://sd-webui-aws-deploy/sd-webui-cf-template.dl1-24xlarge-habana

SD-WebUI will be available using the default port 7860 on the public IP address assigned at provisioning time.

You can delete the resources by running 

aws cloudformation delete-stack --stack-name STACKNAME

### TODO

* Add SSL Encryption to SD-WebUI Server

### Credits

A1111 Project https://github.com/AUTOMATIC1111/stable-diffusion-webui

Heiko Hotz https://towardsdatascience.com/create-your-own-stable-diffusion-ui-on-aws-in-minutes-35480dfcde6a
