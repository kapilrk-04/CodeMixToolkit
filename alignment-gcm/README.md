# Getting alignments from fast-align in GCM docker image

- First step, run:
'''
fast_align -i corpus.f-e -d -v -o -p fwd_params >fwd_align 2>fwd_err
fast_align -i corpus.f-e -r -d -v -o -p rev_params >rev_align 2>rev_err
'''
where corpus.f-e is training corpus, fwd_prarams, fwd_err, rev_params, rev_err are the saved models. Do not remove fwd_error and rev_error!

- Second step:
run:
'''
force_align.py fwd_params fwd_err rev_params rev_err [heuristic] <in.f-e >out.f-e.gdfa
'''

where heuristic is one of: (intersect union grow-diag grow-diag-final grow-diag-final-and) 
    
    default=grow-diag-final-and , 
    in.f-e is the file which you want to get alignment.


# Commands for generating the en-hi aligner

'''
# for training the aligner
!/CodeMixed-Text-Generator/CodeMixed-Text-Generator/alignment_generator/fast_align-master/build/fast_align -i parallel_samanantar_filtered -d -v -o -p fwd_params >fwd_align 2>fwd_err
!/CodeMixed-Text-Generator/CodeMixed-Text-Generator/alignment_generator/fast_align-master/build/fast_align -i parallel_samanantar_filtered -r -d -v -o -p rev_params >rev_align 2>rev_err

# for getting alignments for a query pair of sentences - in.f-e has the input sentences, aligner output in out.f-e.gdfa-sam
/CodeMixed-Text-Generator/CodeMixed-Text-Generator/alignment_generator/fast_align-master/build/force_align.py fwd_params fwd_err rev_params rev_err grow-diag-final-and <in.f-e >out.f-e.gdfa-sam

'''

- You can find the fwd_* and rev_* files in the directory : /CodeMixed-Text-Generator/CodeMixed-Text-Generator/ 