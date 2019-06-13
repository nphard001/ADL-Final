# --------==== main Makefile in ADL final project ====--------
all:
	@echo nothing to build
# ================================================================
# just download original git repo, instead of fork
setup_duplicate:
	git clone https://github.com/nphard001/fashion_retrieval
	rm -rf fashion_retrieval/.git/
	tree -d .
	@echo "NOTES: original .git files are deleted"


