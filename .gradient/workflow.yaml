'on':
  github:
    branches:
      only: main
jobs:
  CloneRepo:
    resources:
      instance-type: C5
    outputs:
      repo:
        type: volume
    uses: git-checkout@v1
    with:
      url: context.event.github.url
  HelloWorld:
    resources:
      instance-type: C5
    needs:
      - CloneRepo
    inputs:
      repo: CloneRepo.outputs.repo
    uses: script@v1
    with:
      script: python /inputs/repo/hello.py
      image: tensorflow/tensorflow:1.14.0-py3
