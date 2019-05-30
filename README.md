# ConvLab
This is a toolkit for developing task-oriented dialog system. We
followed the conventional pipeline framework, where there are 4 seperate
components: NLU, DST, Policy and NLG.

We offer the base class and some SOTA baseline models (coming soon)
for each component. Specially, the NLU, DST and NLG models are trained
individually, while the Policy is trained within a complete pipeline
system in an RL-based manner.

- NLU: Regex method, Seq2seq, JointNLU, ContextualNLU
- DST: Rule, NBT, Sequicity
- Policy: Rule, DQN, MDQN, HRL
- NLG: Templated method, SC-LSTM, CA-LSTM

## Environment

<!---
- Trained NLG model can be downloaded [here](https://www.dropbox.com/s/7d6rr57hmdcz9pd/lstm_tanh_%5B1549590993.11%5D_24_28_1000_0.447.pkl?dl=0).
-->
- Trained NLG model can be downloaded [here](https://www.dropbox.com/s/u1n8jlgr89jnn2f/lstm_tanh_%5B1552674040.43%5D_7_7_400_0.436.pkl?dl=0). 
- Trained NLU model can be downloaded [here](https://www.dropbox.com/s/y2aclsz9t7nmxnr/bi_lstm_%5B1552541377.53%5D_7_7_360_0.912.pkl?dl=0).
- Trained S2S UserSim model can be downloaded [here](https://www.dropbox.com/s/2jxkqp2ad07asps/lstm_%5B1550147645.59%5D_20_29_0.448.p?dl=0).
- Trained MLST NLU model can be downloaded [here](https://1drv.ms/u/s!AmXaP0QRGLFchVZqB047pJdS-tiT). Unzip the downloaded file in the tasktk/nlu/mlst directory. 
- Trained JointNLU model can be downloaded [here](https://1drv.ms/u/s!AmXaP0QRGLFchVn7DNj4s7fghLTo). Unzip the downloaded file in the tasktk/nlu/joint_nlu directory. 

## Document

## To Developer

## How to start
To run the code, you have to first download [mdbt.tar.gz](https://drive.google.com/file/d/1jN8p_PrhgdfBYa2--GqSQiHGFONWuINe/view?usp=sharing)
 then extract it and move the mdbt directory under ./data . The mdbt directory
includes the data and trained model required for building MDBT tracker.

Then, you can just run the ./run.py script to run the dialog on dialog-act level.
Note that the MDBT model receives natural langauge utterances as input, so we used a trivial
rule-based NLG to convert the user response DA into natural langauge format (see tasktk/nlg/template_nlg.py).

The outputs of system policy, MDBT and simulator are logged into ./session.txt, where the turns and sessions
are seperated by separators for clarity.
