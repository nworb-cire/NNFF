# NNFF

This repository contains the code used to train NNFF models for [openpilot](https://github.com/commaai/openpilot), an open-source vehicle automation system.

### What is NNFF?

Neural Network Feed Forward is a model solving the inverse problem of a car's lateral dynamics. Given the current state of the car, it predicts the steering command required to achieve a desired trajectory. This neural network can be used in conjunction with or as a replacement for a traditional controls scheme such as a PID controller.

NNFF was initially brought to openpilot in [version 0.9.6](https://blog.comma.ai/096release/#ml-controls) for the Chevy Bolt after much community pressure. Internal progress seems to have halted, speculation is that this is due to employee attrition. This repository is an attempt to revive the project and bring NNFF to more cars within the official build of openpilot, specifically GM vehicles which have highly nonlinear steering systems and thus stand to benefit the most from a neural network controller.

View development discussion [here](https://discord.com/channels/469524606043160576/1252770070677950606).
