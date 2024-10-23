#!/bin/bash

mvn clean install -U -Dmaven.test.skip=true -T8 -P{{.PROFILE}}

image_repo={{.IMAGE_REPO}}
tag=$(date +%Y%m%d%H%M%S)

docker build -f .Dockerfile -t ${image_repo}:"${tag}" . --platform=linux/amd64
docker login artifacthub.online.cos.jdcloud.com;
docker tag ${image_repo}:"${tag}" artifacthub.online.cos.jdcloud.com/jdap-sys/${image_repo}/${image_repo}:"${tag}"
docker push artifacthub.online.cos.jdcloud.com/jdap-sys/${image_repo}/${image_repo}:"${tag}"
docker rmi artifacthub.online.cos.jdcloud.com/jdap-sys/${image_repo}/${image_repo}:"${tag}"

