import replicate
output = replicate.run(
"web-cardinalblue/controlnet-inpaint-scribble-2:4b7603a6d66cbda55c5b439e0be23d28a07ac1dabfd06da55161c76057ca619d",
    input={
      "prompt": "a cute rabbit sitting on a bench",
      "image": open('./dog-bench-source.png', 'rb'),
      "mask_image": open('./dog-bench-masked-3.png', 'rb'),
      "control_image": open('./dog-bench-scribbled-3.png', 'rb'),
      "num_outputs": 4,
      }
)
print(output)