build-image:
	 docker build -t registry.gitlab.com/lambda-hse/registry/hyperoptworker  .

push-image:
	docker push registry.gitlab.com/lambda-hse/registry/hyperoptworker

run-image:
	docker run -it registry.gitlab.com/lambda-hse/registry/hyperoptworker

debug-image:
	docker run --entrypoint='' -it registry.gitlab.com/lambda-hse/registry/hyperoptworker /bin/bash
