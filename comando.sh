TrainDPGS \
	--train="/home/eraldo/partiallabel/data/dpgs-por/factors.dev.split.1000lines.06.pt.chunks.clause.win1" \
	--templates="/home/eraldo/partiallabel/data/dpgs-por/templates.mcdonald_05_06.ftrs_lemma_postag" \
	--test="/home/eraldo/partiallabel/data/dpgs-por/factors.dev.split.1000lines.06.pt.chunks.clause.win1" \
	--testconll="/home/eraldo/partiallabel/data/dpgs-por/dev.split.1000lines.06.pt.chunk.clauses.pred" \
	--outputconll="/home/eraldo/partiallabel/data/dpgs-por/outputconll" \
	--script="/home/eraldo/rext/experiments/deptree/misc/eval.pl" \
	--numepochs=50 \
	--maxsteps=500 \
	--seed=3 \
	--traincachesize=4294967296 \
	--testcachesize=103863034 \
	--modelfiletosave=model \
	--numthreadtofillweight=2 \
	--alg=pa \
	pernumepoch=15

