def imresize(screen, height, width)
  scope_class = Tensorflow::Scope.new
  input = Const(scope_class, screen.to_a)
  output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('Cast', 'Cast', Hash['DstT' => 1], [input]))
  output = input.operation.g.AddOperation(Tensorflow::OpSpec.new('ExpandDims', 'ExpandDims', nil, [output.output(0), Const(scope_class.subscope('make_batch'), 0, :int32)]))
  output = input.operation.g.AddOperation(Tensorflow::OpSpec.new(
    'ResizeBilinear',
    'ResizeBilinear',
    nil,
    [output.output(0), Const(scope_class.subscope('size'), [height, width], :int32)])
  )
  output = output.output(0)
  graph = scope_class.graph
  session_op = Tensorflow::Session_options.new
  session = Tensorflow::Session.new(graph, session_op)
  out_tensor = session.run({}, [output], [])
  n = N[out_tensor]
  shape = n.shape
  shape.delete(1)
  n.reshape(shape)
end