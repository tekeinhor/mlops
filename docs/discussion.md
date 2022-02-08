# Going further

Here are some elements where we could have gone further.
## Software Eng Improvements
- Dockerize training: packaging the training in docker
- Register a database to MLFlow .
- Increase test coverage.
- Improve the config by applying [twelve-factor app methodology](https://12factor.net/config)
- Smoothen the train-predict process. Allow for instance to perform a train before predict if it was done before.
- Add a docker-compose to ease the train-predict inside docker.

## Data Science Improvements
- Combine different sampling techniques to threshold-moving.
- Use the distribution info given as a prior for the prediction (the distribution of clicks for a given bot tends to be skewed towards `click_ad` and `send_email`).