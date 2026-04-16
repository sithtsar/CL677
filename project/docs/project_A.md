# Projects (total points: 100)
> Important Note: The evaluation is based on a project report, which should include figures anddiscussion of results, as well as an oral viva based on a presentation (conducted online). The following points should be kept in mind while working on the project and preparing the report:
- Any use of ChatGPT and related LLM tools should be declared explicitly, in writing in
the report, and also orally during the viva. It is acceptable to use such tools for obtaining
snippets of code to perform specified tasks, or to discover information on a topic. However,
using entire paragraphs from ChatGPT for your discussion of results, or reproducing the
generated code as-is, is unacceptable.
- It is strongly advised that you begin working on the project early, and contact me or the TA
well before the final viva in case you get stuck.
- Resources to help with the following three projects (papers, books, slides) are available on
the google drive folder shared on moodle.


## Project A: Smooth Random Noise and the Stratonovich Limit
In this project, you will use smooth random functions (Filip, Javved, Trefethen, SIAM Rev.,
61(1), 185, 2017) to represent stochastic noise. With these functions, you can treat the stochastic
equation like a usual ODE and integrate it with standard tools (no need for Stochastic calculus). By
reducing hte correlation time-scale of the function you can approach the Stratonovich interpretation
of white-noise-driven SDE. You will verify this limit in the project for the problem of the geometric
random walk (see Assignment 3).

1. Construct the smooth random functions (do not use the ready-made Chebfun package tools
but write your own code instead to generate the functions) and implement them to model
Brownian motion. Reproduce Figs. 1 and 6 from the review paper.
2. Use the smooth random function as the stochastic noise in the goemetric random walk prob-
lem, where the diffusion coefficient is proportional to the dependent variable (see Assignment
3). Show that when the correlation time of the forcing function is taken to zero, one obtains
the results of the Stratonovich SDE and not the Ito SDE
