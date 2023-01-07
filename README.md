# CMU10-714: Deep Learning System
This repo contains all the learning materials and homework implementation for the CMU10-714: Deep Learning System course.

## Course Overview

Deep learning methods have revolutionized a number of fields in Artificial Intelligence and Machine Learning in recent years. The widespread adoption of deep learning methods have in no small part been driven by the widespread availability of easy-to-use deep learning systems, such as PyTorch and TensorFlow. But despite their widespread availability and use, it is much less common for students to get involved with the internals of these libraries, to understand how they function at a fundamental level. But understanding these libraries deeply will help you make better use of their functionality, and enable you to develop or extend these libraries when needed to fit your own custom use cases in deep learning.

The goal of this course is to provide students an understanding and overview of the “full stack” of deep learning systems, ranging from the high-level modeling design of modern deep learning systems, to the basic implementation of automatic differentiation tools, to the underlying device-level implementation of efficient algorithms. Throughout the course, students will design and build from scratch a complete deep learning library, capable of efficient GPU-based operations, automatic differentiation of all implemented functions, and the necessary modules to support parameterized layers, loss functions, data loaders, and optimizers. Using these tools, students will then build several state-of-the-art modeling methods, including convolutional networks for image classification and segmentation, recurrent networks and self-attention models for sequential tasks such as language modeling, and generative models for image generation.

## Course Materials

- Course Website: <https://dlsyscourse.org/>
- Lecture Record: <https://www.youtube.com/watch?v=qbJqOFMyIwg>
- Lecture Notes: under [lectures](./lectures/) folder.
- Homework Implementation: under [homework](./homework/) folder.

## Final Project

The goal of the final course project is to implement some non-trivial deep learning model or component within the needle framework. The assignment as a whole is fairly open-ended, and many different options are available: you could implement some class of model (requiring functionality beyond what we cover in the homework), some new suite of optimizers, some new functionality within the autodiff framework, or some new backend support for needle.

In our final project, we added Apple M1 chip support to the Needle framework. We passed all the unit tests used in the assignments, compared its performance to the naive CPU backend and trained ResNet on the CIFAR-10 dataset. The implementation is open-sourced [here](https://github.com/wenjunsun/dlsys-needle-m1).

Some reference resources:
- <https://github.com/larsgeb/m1-gpu-cpp>
- <https://larsgeb.github.io/2022/04/20/m1-gpu.html>
- <https://github.com/bkvogel/metal_performance_testing>



## Want to learn more ?

Check out [this repository](https://github.com/PKUFlyingPig/cs-self-learning) which contains all my self-learning materials : )