# Final Project
## To Train Original Repo
1. 
```
cd fashion_retrieval && python train_sl.py --log-interval=50 --lr=0.001  --batch-size=128 --model-folder="models/"
```
2. 
```
cd fashion_retrieval && python train_rl.py --log-interval=10 --lr=0.0001 --top-k=4 --batch-size=128 --tau=0.2 --pretrained-model="models/sl-10.pt"
```

# Commit Messages

常用
+ ADD: 加新功能新module新class
+ ENH: 改進東西（最常用）
+ BUG: 修掉的bug內容
+ MAINT, MISC: 其他內容如說明文件typo或單純不知道要寫什麼 (um...🤔)


> [scipy development workflow](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html)
> Standard acronyms to start the commit message with are:
> + API: an (incompatible) API change
> + BENCH: changes to the benchmark suite
> + BLD: change related to building numpy
> + BUG: bug fix
> + DEP: deprecate something, or remove a deprecated object
> + DEV: development tool or utility
> + DOC: documentation
> + ENH: enhancement
> + MAINT: maintenance commit (refactoring, typos, etc.)
> + REV: revert an earlier commit
> + STY: style fix (whitespace, PEP8)
> + TST: addition or modification of tests
> + REL: related to releasing numpy
