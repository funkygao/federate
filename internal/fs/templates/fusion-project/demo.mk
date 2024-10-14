##@ Demo: Run the demo project on your host

DEMO_MANIFEST := {{.FusionProjectsName}}/demo/manifest.yaml
DEMO_ENV := on-premise
DEMO_SILENT := false

consolidate-demo:confirm-env ## Generate the target system from the demo manifest.
	$(call merge_project,$(DEMO_MANIFEST),{{.FusionProjectsName}}/demo,$(DEMO_ENV),$(DEMO_SILENT))
	echo "👉 Next, cd generated/demo && make run"
