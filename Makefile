.PHONY: data clean lint train

data: data/example_traj_s.csv data/example_traj_x.csv experiments/pws experiments/mlpws experiments/infonce experiments/doe

data/example_traj_s.csv data/example_traj_x.csv:
	uv run scripts/01-ar_dataset.py

experiments/pws:
	uv run scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 -o experiments/pws --estimator PWS

experiments/mlpws:
	uv run scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 -o experiments/mlpws --estimator ML-PWS

experiments/infonce:
	uv run scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 --backward_epochs 100 -o experiments/infonce --estimator InfoNCE

experiments/doe:
	uv run scripts/05-ar_input_mlpws.py --gain 1.0 --ar_std 1.0 --output_noise 0.2 --backward_epochs 100 -o experiments/doe --estimator DoE


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
