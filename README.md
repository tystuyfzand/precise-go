Precise-Lite for Go
===================

This is a Go port of the [MycroftAI Precise](https://github.com/MycroftAI/mycroft-precise) engine, using tflite and onnxruntime 
(based on the [OpenVoiceOS fork](https://github.com/OpenVoiceOS/precise-lite)).

I needed this for my Discord bot [Astra](https://astra.bot) - which I'm currently training a model for. Help us out [here](https://train.astra.bot/) and submit a few voice clips!


What is working?
----------------

Everything should work, though I'm not sure if my port is off in any way. I'm not an AI/Machine Learning expert, I just read the code!

The Dockerfiles contain a pre-built Go image with all dependencies required to use this library. If you want to use it
yourself, you can use a FROM statement for a [to be determined] public image, use it to build, and then copy the tflite
runtime.

Supported Backends
------------------

Currently, I've tested it with tflite and onnxruntime (cpu and gpu) - onnxruntime is incredibly overkill and not
necessary.

Example
-------

See `runner_test.go` - this contains an example usage of both tflite and onnxruntime - though onnxruntime is slower, 
it was more of a test.

Contributions, Bug Reports, etc.
--------------------------------

All of these are welcome! If I made an error in the logic or there's something wrong, please let me know!