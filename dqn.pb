
A
step/step/initial_valueConst*
dtype0*
value	B : 
U
	step/step
VariableV2*
	container *
shared_name *
dtype0*
shape: 

step/step/AssignAssign	step/stepstep/step/initial_value*
validate_shape(*
use_locking(*
T0*
_class
loc:@step/step
L
step/step/readIdentity	step/step*
T0*
_class
loc:@step/step
:
step/step_inputPlaceholder*
shape:*
dtype0

step/AssignAssign	step/stepstep/step_input*
validate_shape(*
use_locking( *
T0*
_class
loc:@step/step
J
main/s_tPlaceholder*$
shape:˙˙˙˙˙˙˙˙˙TT*
dtype0

,main/l1/w/Initializer/truncated_normal/shapeConst*
dtype0*
_class
loc:@main/l1/w*%
valueB"             
v
+main/l1/w/Initializer/truncated_normal/meanConst*
dtype0*
_class
loc:@main/l1/w*
valueB
 *    
x
-main/l1/w/Initializer/truncated_normal/stddevConst*
dtype0*
_class
loc:@main/l1/w*
valueB
 *
×Ł<
Ä
6main/l1/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal,main/l1/w/Initializer/truncated_normal/shape*
seed2
*
T0*
dtype0*
_class
loc:@main/l1/w*

seed{
ż
*main/l1/w/Initializer/truncated_normal/mulMul6main/l1/w/Initializer/truncated_normal/TruncatedNormal-main/l1/w/Initializer/truncated_normal/stddev*
T0*
_class
loc:@main/l1/w
­
&main/l1/w/Initializer/truncated_normalAdd*main/l1/w/Initializer/truncated_normal/mul+main/l1/w/Initializer/truncated_normal/mean*
T0*
_class
loc:@main/l1/w

	main/l1/w
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_class
loc:@main/l1/w

main/l1/w/AssignAssign	main/l1/w&main/l1/w/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l1/w
L
main/l1/w/readIdentity	main/l1/w*
T0*
_class
loc:@main/l1/w

main/l1/Conv2DConv2Dmain/s_tmain/l1/w/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
t
 main/l1/biases/Initializer/ConstConst*
dtype0*!
_class
loc:@main/l1/biases*
valueB *    

main/l1/biases
VariableV2*
	container *
shared_name *
dtype0*
shape: *!
_class
loc:@main/l1/biases
Ś
main/l1/biases/AssignAssignmain/l1/biases main/l1/biases/Initializer/Const*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l1/biases
[
main/l1/biases/readIdentitymain/l1/biases*
T0*!
_class
loc:@main/l1/biases
_
main/l1/BiasAddBiasAddmain/l1/Conv2Dmain/l1/biases/read*
T0*
data_formatNHWC
+
	main/ReluRelumain/l1/BiasAdd*
T0

,main/l2/w/Initializer/truncated_normal/shapeConst*
dtype0*
_class
loc:@main/l2/w*%
valueB"          @   
v
+main/l2/w/Initializer/truncated_normal/meanConst*
dtype0*
_class
loc:@main/l2/w*
valueB
 *    
x
-main/l2/w/Initializer/truncated_normal/stddevConst*
dtype0*
_class
loc:@main/l2/w*
valueB
 *
×Ł<
Ä
6main/l2/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal,main/l2/w/Initializer/truncated_normal/shape*
seed2*
T0*
dtype0*
_class
loc:@main/l2/w*

seed{
ż
*main/l2/w/Initializer/truncated_normal/mulMul6main/l2/w/Initializer/truncated_normal/TruncatedNormal-main/l2/w/Initializer/truncated_normal/stddev*
T0*
_class
loc:@main/l2/w
­
&main/l2/w/Initializer/truncated_normalAdd*main/l2/w/Initializer/truncated_normal/mul+main/l2/w/Initializer/truncated_normal/mean*
T0*
_class
loc:@main/l2/w

	main/l2/w
VariableV2*
	container *
shared_name *
dtype0*
shape: @*
_class
loc:@main/l2/w

main/l2/w/AssignAssign	main/l2/w&main/l2/w/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l2/w
L
main/l2/w/readIdentity	main/l2/w*
T0*
_class
loc:@main/l2/w

main/l2/Conv2DConv2D	main/Relumain/l2/w/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
t
 main/l2/biases/Initializer/ConstConst*
dtype0*!
_class
loc:@main/l2/biases*
valueB@*    

main/l2/biases
VariableV2*
	container *
shared_name *
dtype0*
shape:@*!
_class
loc:@main/l2/biases
Ś
main/l2/biases/AssignAssignmain/l2/biases main/l2/biases/Initializer/Const*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l2/biases
[
main/l2/biases/readIdentitymain/l2/biases*
T0*!
_class
loc:@main/l2/biases
_
main/l2/BiasAddBiasAddmain/l2/Conv2Dmain/l2/biases/read*
T0*
data_formatNHWC
-
main/Relu_1Relumain/l2/BiasAdd*
T0

,main/l3/w/Initializer/truncated_normal/shapeConst*
dtype0*
_class
loc:@main/l3/w*%
valueB"      @   @   
v
+main/l3/w/Initializer/truncated_normal/meanConst*
dtype0*
_class
loc:@main/l3/w*
valueB
 *    
x
-main/l3/w/Initializer/truncated_normal/stddevConst*
dtype0*
_class
loc:@main/l3/w*
valueB
 *
×Ł<
Ä
6main/l3/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal,main/l3/w/Initializer/truncated_normal/shape*
seed2**
T0*
dtype0*
_class
loc:@main/l3/w*

seed{
ż
*main/l3/w/Initializer/truncated_normal/mulMul6main/l3/w/Initializer/truncated_normal/TruncatedNormal-main/l3/w/Initializer/truncated_normal/stddev*
T0*
_class
loc:@main/l3/w
­
&main/l3/w/Initializer/truncated_normalAdd*main/l3/w/Initializer/truncated_normal/mul+main/l3/w/Initializer/truncated_normal/mean*
T0*
_class
loc:@main/l3/w

	main/l3/w
VariableV2*
	container *
shared_name *
dtype0*
shape:@@*
_class
loc:@main/l3/w

main/l3/w/AssignAssign	main/l3/w&main/l3/w/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l3/w
L
main/l3/w/readIdentity	main/l3/w*
T0*
_class
loc:@main/l3/w

main/l3/Conv2DConv2Dmain/Relu_1main/l3/w/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
t
 main/l3/biases/Initializer/ConstConst*
dtype0*!
_class
loc:@main/l3/biases*
valueB@*    

main/l3/biases
VariableV2*
	container *
shared_name *
dtype0*
shape:@*!
_class
loc:@main/l3/biases
Ś
main/l3/biases/AssignAssignmain/l3/biases main/l3/biases/Initializer/Const*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l3/biases
[
main/l3/biases/readIdentitymain/l3/biases*
T0*!
_class
loc:@main/l3/biases
_
main/l3/BiasAddBiasAddmain/l3/Conv2Dmain/l3/biases/read*
T0*
data_formatNHWC
-
main/Relu_2Relumain/l3/BiasAdd*
T0
G
main/Reshape/shapeConst*
dtype0*
valueB"˙˙˙˙@  
O
main/ReshapeReshapemain/Relu_2main/Reshape/shape*
Tshape0*
T0

.main/l4/Matrix/Initializer/random_normal/shapeConst*
dtype0*!
_class
loc:@main/l4/Matrix*
valueB"@     
}
-main/l4/Matrix/Initializer/random_normal/meanConst*
dtype0*!
_class
loc:@main/l4/Matrix*
valueB
 *    

/main/l4/Matrix/Initializer/random_normal/stddevConst*
dtype0*!
_class
loc:@main/l4/Matrix*
valueB
 *
×Ł<
×
=main/l4/Matrix/Initializer/random_normal/RandomStandardNormalRandomStandardNormal.main/l4/Matrix/Initializer/random_normal/shape*
seed2<*
T0*
dtype0*!
_class
loc:@main/l4/Matrix*

seed{
Ď
,main/l4/Matrix/Initializer/random_normal/mulMul=main/l4/Matrix/Initializer/random_normal/RandomStandardNormal/main/l4/Matrix/Initializer/random_normal/stddev*
T0*!
_class
loc:@main/l4/Matrix
¸
(main/l4/Matrix/Initializer/random_normalAdd,main/l4/Matrix/Initializer/random_normal/mul-main/l4/Matrix/Initializer/random_normal/mean*
T0*!
_class
loc:@main/l4/Matrix

main/l4/Matrix
VariableV2*
	container *
shared_name *
dtype0*
shape:
Ŕ*!
_class
loc:@main/l4/Matrix
Ž
main/l4/Matrix/AssignAssignmain/l4/Matrix(main/l4/Matrix/Initializer/random_normal*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l4/Matrix
[
main/l4/Matrix/readIdentitymain/l4/Matrix*
T0*!
_class
loc:@main/l4/Matrix
q
main/l4/bias/Initializer/ConstConst*
dtype0*
_class
loc:@main/l4/bias*
valueB*    
~
main/l4/bias
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_class
loc:@main/l4/bias

main/l4/bias/AssignAssignmain/l4/biasmain/l4/bias/Initializer/Const*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l4/bias
U
main/l4/bias/readIdentitymain/l4/bias*
T0*
_class
loc:@main/l4/bias
j
main/l4/MatMulMatMulmain/Reshapemain/l4/Matrix/read*
T0*
transpose_a( *
transpose_b( 
]
main/l4/BiasAddBiasAddmain/l4/MatMulmain/l4/bias/read*
T0*
data_formatNHWC
.
main/l4/ReluRelumain/l4/BiasAdd*
T0

-main/q/Matrix/Initializer/random_normal/shapeConst*
dtype0* 
_class
loc:@main/q/Matrix*
valueB"      
{
,main/q/Matrix/Initializer/random_normal/meanConst*
dtype0* 
_class
loc:@main/q/Matrix*
valueB
 *    
}
.main/q/Matrix/Initializer/random_normal/stddevConst*
dtype0* 
_class
loc:@main/q/Matrix*
valueB
 *
×Ł<
Ô
<main/q/Matrix/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-main/q/Matrix/Initializer/random_normal/shape*
seed2L*
T0*
dtype0* 
_class
loc:@main/q/Matrix*

seed{
Ë
+main/q/Matrix/Initializer/random_normal/mulMul<main/q/Matrix/Initializer/random_normal/RandomStandardNormal.main/q/Matrix/Initializer/random_normal/stddev*
T0* 
_class
loc:@main/q/Matrix
´
'main/q/Matrix/Initializer/random_normalAdd+main/q/Matrix/Initializer/random_normal/mul,main/q/Matrix/Initializer/random_normal/mean*
T0* 
_class
loc:@main/q/Matrix

main/q/Matrix
VariableV2*
	container *
shared_name *
dtype0*
shape:	* 
_class
loc:@main/q/Matrix
Ş
main/q/Matrix/AssignAssignmain/q/Matrix'main/q/Matrix/Initializer/random_normal*
validate_shape(*
use_locking(*
T0* 
_class
loc:@main/q/Matrix
X
main/q/Matrix/readIdentitymain/q/Matrix*
T0* 
_class
loc:@main/q/Matrix
n
main/q/bias/Initializer/ConstConst*
dtype0*
_class
loc:@main/q/bias*
valueB*    
{
main/q/bias
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_class
loc:@main/q/bias

main/q/bias/AssignAssignmain/q/biasmain/q/bias/Initializer/Const*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/q/bias
R
main/q/bias/readIdentitymain/q/bias*
T0*
_class
loc:@main/q/bias
h
main/q/MatMulMatMulmain/l4/Relumain/q/Matrix/read*
T0*
transpose_a( *
transpose_b( 
Z
main/q/BiasAddBiasAddmain/q/MatMulmain/q/bias/read*
T0*
data_formatNHWC
?
main/ArgMax/dimensionConst*
dtype0*
value	B :
d
main/ArgMaxArgMaxmain/q/BiasAddmain/ArgMax/dimension*
T0*

Tidx0*
output_type0	
E
main/Mean/reduction_indicesConst*
dtype0*
value	B : 
d
	main/MeanMeanmain/q/BiasAddmain/Mean/reduction_indices*
T0*
	keep_dims( *

Tidx0
F
main/strided_slice/stackConst*
dtype0*
valueB: 
H
main/strided_slice/stack_1Const*
dtype0*
valueB:
H
main/strided_slice/stack_2Const*
dtype0*
valueB:
ů
main/strided_sliceStridedSlice	main/Meanmain/strided_slice/stackmain/strided_slice/stack_1main/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
end_mask *
Index0*
T0*
ellipsis_mask *
new_axis_mask 
=
main/q/0/tagConst*
dtype0*
valueB Bmain/q/0
G
main/q/0HistogramSummarymain/q/0/tagmain/strided_slice*
T0
H
main/strided_slice_1/stackConst*
dtype0*
valueB:
J
main/strided_slice_1/stack_1Const*
dtype0*
valueB:
J
main/strided_slice_1/stack_2Const*
dtype0*
valueB:

main/strided_slice_1StridedSlice	main/Meanmain/strided_slice_1/stackmain/strided_slice_1/stack_1main/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
end_mask *
Index0*
T0*
ellipsis_mask *
new_axis_mask 
=
main/q/1/tagConst*
dtype0*
valueB Bmain/q/1
I
main/q/1HistogramSummarymain/q/1/tagmain/strided_slice_1*
T0
H
main/strided_slice_2/stackConst*
dtype0*
valueB:
J
main/strided_slice_2/stack_1Const*
dtype0*
valueB:
J
main/strided_slice_2/stack_2Const*
dtype0*
valueB:

main/strided_slice_2StridedSlice	main/Meanmain/strided_slice_2/stackmain/strided_slice_2/stack_1main/strided_slice_2/stack_2*
shrink_axis_mask*

begin_mask *
end_mask *
Index0*
T0*
ellipsis_mask *
new_axis_mask 
=
main/q/2/tagConst*
dtype0*
valueB Bmain/q/2
I
main/q/2HistogramSummarymain/q/2/tagmain/strided_slice_2*
T0
H
main/strided_slice_3/stackConst*
dtype0*
valueB:
J
main/strided_slice_3/stack_1Const*
dtype0*
valueB:
J
main/strided_slice_3/stack_2Const*
dtype0*
valueB:

main/strided_slice_3StridedSlice	main/Meanmain/strided_slice_3/stackmain/strided_slice_3/stack_1main/strided_slice_3/stack_2*
shrink_axis_mask*

begin_mask *
end_mask *
Index0*
T0*
ellipsis_mask *
new_axis_mask 
=
main/q/3/tagConst*
dtype0*
valueB Bmain/q/3
I
main/q/3HistogramSummarymain/q/3/tagmain/strided_slice_3*
T0
X
main/Merge/MergeSummaryMergeSummarymain/q/0main/q/1main/q/2main/q/3*
N
S
target/target_s_tPlaceholder*$
shape:˙˙˙˙˙˙˙˙˙TT*
dtype0

5target/target_l1/w/Initializer/truncated_normal/shapeConst*
dtype0*%
_class
loc:@target/target_l1/w*%
valueB"             

4target/target_l1/w/Initializer/truncated_normal/meanConst*
dtype0*%
_class
loc:@target/target_l1/w*
valueB
 *    

6target/target_l1/w/Initializer/truncated_normal/stddevConst*
dtype0*%
_class
loc:@target/target_l1/w*
valueB
 *
×Ł<
ß
?target/target_l1/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5target/target_l1/w/Initializer/truncated_normal/shape*
seed2y*
T0*
dtype0*%
_class
loc:@target/target_l1/w*

seed{
ă
3target/target_l1/w/Initializer/truncated_normal/mulMul?target/target_l1/w/Initializer/truncated_normal/TruncatedNormal6target/target_l1/w/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@target/target_l1/w
Ń
/target/target_l1/w/Initializer/truncated_normalAdd3target/target_l1/w/Initializer/truncated_normal/mul4target/target_l1/w/Initializer/truncated_normal/mean*
T0*%
_class
loc:@target/target_l1/w

target/target_l1/w
VariableV2*
	container *
shared_name *
dtype0*
shape: *%
_class
loc:@target/target_l1/w
Á
target/target_l1/w/AssignAssigntarget/target_l1/w/target/target_l1/w/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*%
_class
loc:@target/target_l1/w
g
target/target_l1/w/readIdentitytarget/target_l1/w*
T0*%
_class
loc:@target/target_l1/w
Ž
target/target_l1/Conv2DConv2Dtarget/target_s_ttarget/target_l1/w/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

)target/target_l1/biases/Initializer/ConstConst*
dtype0**
_class 
loc:@target/target_l1/biases*
valueB *    

target/target_l1/biases
VariableV2*
	container *
shared_name *
dtype0*
shape: **
_class 
loc:@target/target_l1/biases
Ę
target/target_l1/biases/AssignAssigntarget/target_l1/biases)target/target_l1/biases/Initializer/Const*
validate_shape(*
use_locking(*
T0**
_class 
loc:@target/target_l1/biases
v
target/target_l1/biases/readIdentitytarget/target_l1/biases*
T0**
_class 
loc:@target/target_l1/biases
z
target/target_l1/BiasAddBiasAddtarget/target_l1/Conv2Dtarget/target_l1/biases/read*
T0*
data_formatNHWC
6
target/ReluRelutarget/target_l1/BiasAdd*
T0

5target/target_l2/w/Initializer/truncated_normal/shapeConst*
dtype0*%
_class
loc:@target/target_l2/w*%
valueB"          @   

4target/target_l2/w/Initializer/truncated_normal/meanConst*
dtype0*%
_class
loc:@target/target_l2/w*
valueB
 *    

6target/target_l2/w/Initializer/truncated_normal/stddevConst*
dtype0*%
_class
loc:@target/target_l2/w*
valueB
 *
×Ł<
ŕ
?target/target_l2/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5target/target_l2/w/Initializer/truncated_normal/shape*
seed2*
T0*
dtype0*%
_class
loc:@target/target_l2/w*

seed{
ă
3target/target_l2/w/Initializer/truncated_normal/mulMul?target/target_l2/w/Initializer/truncated_normal/TruncatedNormal6target/target_l2/w/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@target/target_l2/w
Ń
/target/target_l2/w/Initializer/truncated_normalAdd3target/target_l2/w/Initializer/truncated_normal/mul4target/target_l2/w/Initializer/truncated_normal/mean*
T0*%
_class
loc:@target/target_l2/w

target/target_l2/w
VariableV2*
	container *
shared_name *
dtype0*
shape: @*%
_class
loc:@target/target_l2/w
Á
target/target_l2/w/AssignAssigntarget/target_l2/w/target/target_l2/w/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*%
_class
loc:@target/target_l2/w
g
target/target_l2/w/readIdentitytarget/target_l2/w*
T0*%
_class
loc:@target/target_l2/w
¨
target/target_l2/Conv2DConv2Dtarget/Relutarget/target_l2/w/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

)target/target_l2/biases/Initializer/ConstConst*
dtype0**
_class 
loc:@target/target_l2/biases*
valueB@*    

target/target_l2/biases
VariableV2*
	container *
shared_name *
dtype0*
shape:@**
_class 
loc:@target/target_l2/biases
Ę
target/target_l2/biases/AssignAssigntarget/target_l2/biases)target/target_l2/biases/Initializer/Const*
validate_shape(*
use_locking(*
T0**
_class 
loc:@target/target_l2/biases
v
target/target_l2/biases/readIdentitytarget/target_l2/biases*
T0**
_class 
loc:@target/target_l2/biases
z
target/target_l2/BiasAddBiasAddtarget/target_l2/Conv2Dtarget/target_l2/biases/read*
T0*
data_formatNHWC
8
target/Relu_1Relutarget/target_l2/BiasAdd*
T0

5target/target_l3/w/Initializer/truncated_normal/shapeConst*
dtype0*%
_class
loc:@target/target_l3/w*%
valueB"      @   @   

4target/target_l3/w/Initializer/truncated_normal/meanConst*
dtype0*%
_class
loc:@target/target_l3/w*
valueB
 *    

6target/target_l3/w/Initializer/truncated_normal/stddevConst*
dtype0*%
_class
loc:@target/target_l3/w*
valueB
 *
×Ł<
ŕ
?target/target_l3/w/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5target/target_l3/w/Initializer/truncated_normal/shape*
seed2*
T0*
dtype0*%
_class
loc:@target/target_l3/w*

seed{
ă
3target/target_l3/w/Initializer/truncated_normal/mulMul?target/target_l3/w/Initializer/truncated_normal/TruncatedNormal6target/target_l3/w/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@target/target_l3/w
Ń
/target/target_l3/w/Initializer/truncated_normalAdd3target/target_l3/w/Initializer/truncated_normal/mul4target/target_l3/w/Initializer/truncated_normal/mean*
T0*%
_class
loc:@target/target_l3/w

target/target_l3/w
VariableV2*
	container *
shared_name *
dtype0*
shape:@@*%
_class
loc:@target/target_l3/w
Á
target/target_l3/w/AssignAssigntarget/target_l3/w/target/target_l3/w/Initializer/truncated_normal*
validate_shape(*
use_locking(*
T0*%
_class
loc:@target/target_l3/w
g
target/target_l3/w/readIdentitytarget/target_l3/w*
T0*%
_class
loc:@target/target_l3/w
Ş
target/target_l3/Conv2DConv2Dtarget/Relu_1target/target_l3/w/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID

)target/target_l3/biases/Initializer/ConstConst*
dtype0**
_class 
loc:@target/target_l3/biases*
valueB@*    

target/target_l3/biases
VariableV2*
	container *
shared_name *
dtype0*
shape:@**
_class 
loc:@target/target_l3/biases
Ę
target/target_l3/biases/AssignAssigntarget/target_l3/biases)target/target_l3/biases/Initializer/Const*
validate_shape(*
use_locking(*
T0**
_class 
loc:@target/target_l3/biases
v
target/target_l3/biases/readIdentitytarget/target_l3/biases*
T0**
_class 
loc:@target/target_l3/biases
z
target/target_l3/BiasAddBiasAddtarget/target_l3/Conv2Dtarget/target_l3/biases/read*
T0*
data_formatNHWC
8
target/Relu_2Relutarget/target_l3/BiasAdd*
T0
I
target/Reshape/shapeConst*
dtype0*
valueB"˙˙˙˙@  
U
target/ReshapeReshapetarget/Relu_2target/Reshape/shape*
Tshape0*
T0

7target/target_l4/Matrix/Initializer/random_normal/shapeConst*
dtype0**
_class 
loc:@target/target_l4/Matrix*
valueB"@     

6target/target_l4/Matrix/Initializer/random_normal/meanConst*
dtype0**
_class 
loc:@target/target_l4/Matrix*
valueB
 *    

8target/target_l4/Matrix/Initializer/random_normal/stddevConst*
dtype0**
_class 
loc:@target/target_l4/Matrix*
valueB
 *
×Ł<
ó
Ftarget/target_l4/Matrix/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7target/target_l4/Matrix/Initializer/random_normal/shape*
seed2Ť*
T0*
dtype0**
_class 
loc:@target/target_l4/Matrix*

seed{
ó
5target/target_l4/Matrix/Initializer/random_normal/mulMulFtarget/target_l4/Matrix/Initializer/random_normal/RandomStandardNormal8target/target_l4/Matrix/Initializer/random_normal/stddev*
T0**
_class 
loc:@target/target_l4/Matrix
Ü
1target/target_l4/Matrix/Initializer/random_normalAdd5target/target_l4/Matrix/Initializer/random_normal/mul6target/target_l4/Matrix/Initializer/random_normal/mean*
T0**
_class 
loc:@target/target_l4/Matrix

target/target_l4/Matrix
VariableV2*
	container *
shared_name *
dtype0*
shape:
Ŕ**
_class 
loc:@target/target_l4/Matrix
Ň
target/target_l4/Matrix/AssignAssigntarget/target_l4/Matrix1target/target_l4/Matrix/Initializer/random_normal*
validate_shape(*
use_locking(*
T0**
_class 
loc:@target/target_l4/Matrix
v
target/target_l4/Matrix/readIdentitytarget/target_l4/Matrix*
T0**
_class 
loc:@target/target_l4/Matrix

'target/target_l4/bias/Initializer/ConstConst*
dtype0*(
_class
loc:@target/target_l4/bias*
valueB*    

target/target_l4/bias
VariableV2*
	container *
shared_name *
dtype0*
shape:*(
_class
loc:@target/target_l4/bias
Â
target/target_l4/bias/AssignAssigntarget/target_l4/bias'target/target_l4/bias/Initializer/Const*
validate_shape(*
use_locking(*
T0*(
_class
loc:@target/target_l4/bias
p
target/target_l4/bias/readIdentitytarget/target_l4/bias*
T0*(
_class
loc:@target/target_l4/bias
~
target/target_l4/MatMulMatMultarget/Reshapetarget/target_l4/Matrix/read*
T0*
transpose_a( *
transpose_b( 
x
target/target_l4/BiasAddBiasAddtarget/target_l4/MatMultarget/target_l4/bias/read*
T0*
data_formatNHWC
@
target/target_l4/ReluRelutarget/target_l4/BiasAdd*
T0

6target/target_q/Matrix/Initializer/random_normal/shapeConst*
dtype0*)
_class
loc:@target/target_q/Matrix*
valueB"      

5target/target_q/Matrix/Initializer/random_normal/meanConst*
dtype0*)
_class
loc:@target/target_q/Matrix*
valueB
 *    

7target/target_q/Matrix/Initializer/random_normal/stddevConst*
dtype0*)
_class
loc:@target/target_q/Matrix*
valueB
 *
×Ł<
đ
Etarget/target_q/Matrix/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6target/target_q/Matrix/Initializer/random_normal/shape*
seed2ť*
T0*
dtype0*)
_class
loc:@target/target_q/Matrix*

seed{
ď
4target/target_q/Matrix/Initializer/random_normal/mulMulEtarget/target_q/Matrix/Initializer/random_normal/RandomStandardNormal7target/target_q/Matrix/Initializer/random_normal/stddev*
T0*)
_class
loc:@target/target_q/Matrix
Ř
0target/target_q/Matrix/Initializer/random_normalAdd4target/target_q/Matrix/Initializer/random_normal/mul5target/target_q/Matrix/Initializer/random_normal/mean*
T0*)
_class
loc:@target/target_q/Matrix

target/target_q/Matrix
VariableV2*
	container *
shared_name *
dtype0*
shape:	*)
_class
loc:@target/target_q/Matrix
Î
target/target_q/Matrix/AssignAssigntarget/target_q/Matrix0target/target_q/Matrix/Initializer/random_normal*
validate_shape(*
use_locking(*
T0*)
_class
loc:@target/target_q/Matrix
s
target/target_q/Matrix/readIdentitytarget/target_q/Matrix*
T0*)
_class
loc:@target/target_q/Matrix

&target/target_q/bias/Initializer/ConstConst*
dtype0*'
_class
loc:@target/target_q/bias*
valueB*    

target/target_q/bias
VariableV2*
	container *
shared_name *
dtype0*
shape:*'
_class
loc:@target/target_q/bias
ž
target/target_q/bias/AssignAssigntarget/target_q/bias&target/target_q/bias/Initializer/Const*
validate_shape(*
use_locking(*
T0*'
_class
loc:@target/target_q/bias
m
target/target_q/bias/readIdentitytarget/target_q/bias*
T0*'
_class
loc:@target/target_q/bias

target/target_q/MatMulMatMultarget/target_l4/Relutarget/target_q/Matrix/read*
T0*
transpose_a( *
transpose_b( 
u
target/target_q/BiasAddBiasAddtarget/target_q/MatMultarget/target_q/bias/read*
T0*
data_formatNHWC
U
target/outputs_idxPlaceholder*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0
g
target/GatherNdGatherNdtarget/target_q/BiasAddtarget/outputs_idx*
Tparams0*
Tindices0
?
pred_to_target/q_bPlaceholder*
shape:*
dtype0
¤
pred_to_target/AssignAssigntarget/target_q/biaspred_to_target/q_b*
validate_shape(*
use_locking( *
T0*'
_class
loc:@target/target_q/bias
@
pred_to_target/l3_bPlaceholder*
shape:@*
dtype0
­
pred_to_target/Assign_1Assigntarget/target_l3/biasespred_to_target/l3_b*
validate_shape(*
use_locking( *
T0**
_class 
loc:@target/target_l3/biases
L
pred_to_target/l3_wPlaceholder*
shape:@@*
dtype0
Ł
pred_to_target/Assign_2Assigntarget/target_l3/wpred_to_target/l3_w*
validate_shape(*
use_locking( *
T0*%
_class
loc:@target/target_l3/w
L
pred_to_target/l2_wPlaceholder*
shape: @*
dtype0
Ł
pred_to_target/Assign_3Assigntarget/target_l2/wpred_to_target/l2_w*
validate_shape(*
use_locking( *
T0*%
_class
loc:@target/target_l2/w
@
pred_to_target/l1_bPlaceholder*
shape: *
dtype0
­
pred_to_target/Assign_4Assigntarget/target_l1/biasespred_to_target/l1_b*
validate_shape(*
use_locking( *
T0**
_class 
loc:@target/target_l1/biases
@
pred_to_target/l2_bPlaceholder*
shape:@*
dtype0
­
pred_to_target/Assign_5Assigntarget/target_l2/biasespred_to_target/l2_b*
validate_shape(*
use_locking( *
T0**
_class 
loc:@target/target_l2/biases
D
pred_to_target/q_wPlaceholder*
shape:	*
dtype0
Ş
pred_to_target/Assign_6Assigntarget/target_q/Matrixpred_to_target/q_w*
validate_shape(*
use_locking( *
T0*)
_class
loc:@target/target_q/Matrix
F
pred_to_target/l4_wPlaceholder*
shape:
Ŕ*
dtype0
­
pred_to_target/Assign_7Assigntarget/target_l4/Matrixpred_to_target/l4_w*
validate_shape(*
use_locking( *
T0**
_class 
loc:@target/target_l4/Matrix
L
pred_to_target/l1_wPlaceholder*
shape: *
dtype0
Ł
pred_to_target/Assign_8Assigntarget/target_l1/wpred_to_target/l1_w*
validate_shape(*
use_locking( *
T0*%
_class
loc:@target/target_l1/w
A
pred_to_target/l4_bPlaceholder*
shape:*
dtype0
Š
pred_to_target/Assign_9Assigntarget/target_l4/biaspred_to_target/l4_b*
validate_shape(*
use_locking( *
T0*(
_class
loc:@target/target_l4/bias
J
optimizer/target_q_tPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
F
optimizer/actionPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0	
J
optimizer/action_onehot/ConstConst*
dtype0*
valueB
 *  ?
L
optimizer/action_onehot/Const_1Const*
dtype0*
valueB
 *    
G
optimizer/action_onehot/depthConst*
dtype0*
value	B :
M
 optimizer/action_onehot/on_valueConst*
dtype0*
valueB
 *  ?
N
!optimizer/action_onehot/off_valueConst*
dtype0*
valueB
 *    
ż
optimizer/action_onehotOneHotoptimizer/actionoptimizer/action_onehot/depth optimizer/action_onehot/on_value!optimizer/action_onehot/off_value*
TI0	*
T0*
axis˙˙˙˙˙˙˙˙˙
F
optimizer/mulMulmain/q/BiasAddoptimizer/action_onehot*
T0
G
optimizer/Q/reduction_indicesConst*
dtype0*
value	B :
f
optimizer/QSumoptimizer/muloptimizer/Q/reduction_indices*
T0*
	keep_dims( *

Tidx0
@
optimizer/subSuboptimizer/target_q_toptimizer/Q*
T0
,
optimizer/AbsAbsoptimizer/sub*
T0
=
optimizer/Less/yConst*
dtype0*
valueB
 *  ?
@
optimizer/LessLessoptimizer/Absoptimizer/Less/y*
T0
2
optimizer/SquareSquareoptimizer/sub*
T0
>
optimizer/mul_1/xConst*
dtype0*
valueB
 *   ?
D
optimizer/mul_1Muloptimizer/mul_1/xoptimizer/Square*
T0
.
optimizer/Abs_1Absoptimizer/sub*
T0
>
optimizer/sub_1/yConst*
dtype0*
valueB
 *   ?
C
optimizer/sub_1Suboptimizer/Abs_1optimizer/sub_1/y*
T0
U
optimizer/SelectSelectoptimizer/Lessoptimizer/mul_1optimizer/sub_1*
T0
=
optimizer/ConstConst*
dtype0*
valueB: 
_
optimizer/lossMeanoptimizer/Selectoptimizer/Const*
T0*
	keep_dims( *

Tidx0
G
optimizer/learning_rate_stepPlaceholder*
shape:*
dtype0	
U
(optimizer/ExponentialDecay/learning_rateConst*
dtype0*
valueB
 *o9
]
optimizer/ExponentialDecay/CastCastoptimizer/learning_rate_step*

DstT0*

SrcT0	
N
#optimizer/ExponentialDecay/Cast_1/xConst*
dtype0*
value
B :ô
f
!optimizer/ExponentialDecay/Cast_1Cast#optimizer/ExponentialDecay/Cast_1/x*

DstT0*

SrcT0
P
#optimizer/ExponentialDecay/Cast_2/xConst*
dtype0*
valueB
 *Âu?
z
"optimizer/ExponentialDecay/truedivRealDivoptimizer/ExponentialDecay/Cast!optimizer/ExponentialDecay/Cast_1*
T0
V
 optimizer/ExponentialDecay/FloorFloor"optimizer/ExponentialDecay/truediv*
T0
u
optimizer/ExponentialDecay/PowPow#optimizer/ExponentialDecay/Cast_2/x optimizer/ExponentialDecay/Floor*
T0
t
optimizer/ExponentialDecayMul(optimizer/ExponentialDecay/learning_rateoptimizer/ExponentialDecay/Pow*
T0
@
optimizer/Maximum/xConst*
dtype0*
valueB
 *o9
V
optimizer/MaximumMaximumoptimizer/Maximum/xoptimizer/ExponentialDecay*
T0
B
optimizer/gradients/ShapeConst*
dtype0*
valueB 
F
optimizer/gradients/ConstConst*
dtype0*
valueB
 *  ?
_
optimizer/gradients/FillFilloptimizer/gradients/Shapeoptimizer/gradients/Const*
T0
c
5optimizer/gradients/optimizer/loss_grad/Reshape/shapeConst*
dtype0*
valueB:
˘
/optimizer/gradients/optimizer/loss_grad/ReshapeReshapeoptimizer/gradients/Fill5optimizer/gradients/optimizer/loss_grad/Reshape/shape*
Tshape0*
T0
a
-optimizer/gradients/optimizer/loss_grad/ShapeShapeoptimizer/Select*
out_type0*
T0
Ż
,optimizer/gradients/optimizer/loss_grad/TileTile/optimizer/gradients/optimizer/loss_grad/Reshape-optimizer/gradients/optimizer/loss_grad/Shape*

Tmultiples0*
T0
c
/optimizer/gradients/optimizer/loss_grad/Shape_1Shapeoptimizer/Select*
out_type0*
T0
X
/optimizer/gradients/optimizer/loss_grad/Shape_2Const*
dtype0*
valueB 
[
-optimizer/gradients/optimizer/loss_grad/ConstConst*
dtype0*
valueB: 
ş
,optimizer/gradients/optimizer/loss_grad/ProdProd/optimizer/gradients/optimizer/loss_grad/Shape_1-optimizer/gradients/optimizer/loss_grad/Const*
T0*
	keep_dims( *

Tidx0
]
/optimizer/gradients/optimizer/loss_grad/Const_1Const*
dtype0*
valueB: 
ž
.optimizer/gradients/optimizer/loss_grad/Prod_1Prod/optimizer/gradients/optimizer/loss_grad/Shape_2/optimizer/gradients/optimizer/loss_grad/Const_1*
T0*
	keep_dims( *

Tidx0
[
1optimizer/gradients/optimizer/loss_grad/Maximum/yConst*
dtype0*
value	B :
Ś
/optimizer/gradients/optimizer/loss_grad/MaximumMaximum.optimizer/gradients/optimizer/loss_grad/Prod_11optimizer/gradients/optimizer/loss_grad/Maximum/y*
T0
¤
0optimizer/gradients/optimizer/loss_grad/floordivFloorDiv,optimizer/gradients/optimizer/loss_grad/Prod/optimizer/gradients/optimizer/loss_grad/Maximum*
T0
~
,optimizer/gradients/optimizer/loss_grad/CastCast0optimizer/gradients/optimizer/loss_grad/floordiv*

DstT0*

SrcT0

/optimizer/gradients/optimizer/loss_grad/truedivRealDiv,optimizer/gradients/optimizer/loss_grad/Tile,optimizer/gradients/optimizer/loss_grad/Cast*
T0
[
4optimizer/gradients/optimizer/Select_grad/zeros_like	ZerosLikeoptimizer/mul_1*
T0
ş
0optimizer/gradients/optimizer/Select_grad/SelectSelectoptimizer/Less/optimizer/gradients/optimizer/loss_grad/truediv4optimizer/gradients/optimizer/Select_grad/zeros_like*
T0
ź
2optimizer/gradients/optimizer/Select_grad/Select_1Selectoptimizer/Less4optimizer/gradients/optimizer/Select_grad/zeros_like/optimizer/gradients/optimizer/loss_grad/truediv*
T0
Ş
:optimizer/gradients/optimizer/Select_grad/tuple/group_depsNoOp1^optimizer/gradients/optimizer/Select_grad/Select3^optimizer/gradients/optimizer/Select_grad/Select_1

Boptimizer/gradients/optimizer/Select_grad/tuple/control_dependencyIdentity0optimizer/gradients/optimizer/Select_grad/Select;^optimizer/gradients/optimizer/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimizer/gradients/optimizer/Select_grad/Select

Doptimizer/gradients/optimizer/Select_grad/tuple/control_dependency_1Identity2optimizer/gradients/optimizer/Select_grad/Select_1;^optimizer/gradients/optimizer/Select_grad/tuple/group_deps*
T0*E
_class;
97loc:@optimizer/gradients/optimizer/Select_grad/Select_1
W
.optimizer/gradients/optimizer/mul_1_grad/ShapeConst*
dtype0*
valueB 
d
0optimizer/gradients/optimizer/mul_1_grad/Shape_1Shapeoptimizer/Square*
out_type0*
T0
Â
>optimizer/gradients/optimizer/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs.optimizer/gradients/optimizer/mul_1_grad/Shape0optimizer/gradients/optimizer/mul_1_grad/Shape_1*
T0

,optimizer/gradients/optimizer/mul_1_grad/mulMulBoptimizer/gradients/optimizer/Select_grad/tuple/control_dependencyoptimizer/Square*
T0
Ç
,optimizer/gradients/optimizer/mul_1_grad/SumSum,optimizer/gradients/optimizer/mul_1_grad/mul>optimizer/gradients/optimizer/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
°
0optimizer/gradients/optimizer/mul_1_grad/ReshapeReshape,optimizer/gradients/optimizer/mul_1_grad/Sum.optimizer/gradients/optimizer/mul_1_grad/Shape*
Tshape0*
T0

.optimizer/gradients/optimizer/mul_1_grad/mul_1Muloptimizer/mul_1/xBoptimizer/gradients/optimizer/Select_grad/tuple/control_dependency*
T0
Í
.optimizer/gradients/optimizer/mul_1_grad/Sum_1Sum.optimizer/gradients/optimizer/mul_1_grad/mul_1@optimizer/gradients/optimizer/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
ś
2optimizer/gradients/optimizer/mul_1_grad/Reshape_1Reshape.optimizer/gradients/optimizer/mul_1_grad/Sum_10optimizer/gradients/optimizer/mul_1_grad/Shape_1*
Tshape0*
T0
Š
9optimizer/gradients/optimizer/mul_1_grad/tuple/group_depsNoOp1^optimizer/gradients/optimizer/mul_1_grad/Reshape3^optimizer/gradients/optimizer/mul_1_grad/Reshape_1

Aoptimizer/gradients/optimizer/mul_1_grad/tuple/control_dependencyIdentity0optimizer/gradients/optimizer/mul_1_grad/Reshape:^optimizer/gradients/optimizer/mul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimizer/gradients/optimizer/mul_1_grad/Reshape

Coptimizer/gradients/optimizer/mul_1_grad/tuple/control_dependency_1Identity2optimizer/gradients/optimizer/mul_1_grad/Reshape_1:^optimizer/gradients/optimizer/mul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@optimizer/gradients/optimizer/mul_1_grad/Reshape_1
a
.optimizer/gradients/optimizer/sub_1_grad/ShapeShapeoptimizer/Abs_1*
out_type0*
T0
Y
0optimizer/gradients/optimizer/sub_1_grad/Shape_1Const*
dtype0*
valueB 
Â
>optimizer/gradients/optimizer/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs.optimizer/gradients/optimizer/sub_1_grad/Shape0optimizer/gradients/optimizer/sub_1_grad/Shape_1*
T0
ß
,optimizer/gradients/optimizer/sub_1_grad/SumSumDoptimizer/gradients/optimizer/Select_grad/tuple/control_dependency_1>optimizer/gradients/optimizer/sub_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
°
0optimizer/gradients/optimizer/sub_1_grad/ReshapeReshape,optimizer/gradients/optimizer/sub_1_grad/Sum.optimizer/gradients/optimizer/sub_1_grad/Shape*
Tshape0*
T0
ă
.optimizer/gradients/optimizer/sub_1_grad/Sum_1SumDoptimizer/gradients/optimizer/Select_grad/tuple/control_dependency_1@optimizer/gradients/optimizer/sub_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
l
,optimizer/gradients/optimizer/sub_1_grad/NegNeg.optimizer/gradients/optimizer/sub_1_grad/Sum_1*
T0
´
2optimizer/gradients/optimizer/sub_1_grad/Reshape_1Reshape,optimizer/gradients/optimizer/sub_1_grad/Neg0optimizer/gradients/optimizer/sub_1_grad/Shape_1*
Tshape0*
T0
Š
9optimizer/gradients/optimizer/sub_1_grad/tuple/group_depsNoOp1^optimizer/gradients/optimizer/sub_1_grad/Reshape3^optimizer/gradients/optimizer/sub_1_grad/Reshape_1

Aoptimizer/gradients/optimizer/sub_1_grad/tuple/control_dependencyIdentity0optimizer/gradients/optimizer/sub_1_grad/Reshape:^optimizer/gradients/optimizer/sub_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimizer/gradients/optimizer/sub_1_grad/Reshape

Coptimizer/gradients/optimizer/sub_1_grad/tuple/control_dependency_1Identity2optimizer/gradients/optimizer/sub_1_grad/Reshape_1:^optimizer/gradients/optimizer/sub_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@optimizer/gradients/optimizer/sub_1_grad/Reshape_1
˘
/optimizer/gradients/optimizer/Square_grad/mul/xConstD^optimizer/gradients/optimizer/mul_1_grad/tuple/control_dependency_1*
dtype0*
valueB
 *   @
}
-optimizer/gradients/optimizer/Square_grad/mulMul/optimizer/gradients/optimizer/Square_grad/mul/xoptimizer/sub*
T0
ł
/optimizer/gradients/optimizer/Square_grad/mul_1MulCoptimizer/gradients/optimizer/mul_1_grad/tuple/control_dependency_1-optimizer/gradients/optimizer/Square_grad/mul*
T0
M
-optimizer/gradients/optimizer/Abs_1_grad/SignSignoptimizer/sub*
T0
Ž
,optimizer/gradients/optimizer/Abs_1_grad/mulMulAoptimizer/gradients/optimizer/sub_1_grad/tuple/control_dependency-optimizer/gradients/optimizer/Abs_1_grad/Sign*
T0
Ő
optimizer/gradients/AddNAddN/optimizer/gradients/optimizer/Square_grad/mul_1,optimizer/gradients/optimizer/Abs_1_grad/mul*
T0*B
_class8
64loc:@optimizer/gradients/optimizer/Square_grad/mul_1*
N
d
,optimizer/gradients/optimizer/sub_grad/ShapeShapeoptimizer/target_q_t*
out_type0*
T0
]
.optimizer/gradients/optimizer/sub_grad/Shape_1Shapeoptimizer/Q*
out_type0*
T0
ź
<optimizer/gradients/optimizer/sub_grad/BroadcastGradientArgsBroadcastGradientArgs,optimizer/gradients/optimizer/sub_grad/Shape.optimizer/gradients/optimizer/sub_grad/Shape_1*
T0
Ż
*optimizer/gradients/optimizer/sub_grad/SumSumoptimizer/gradients/AddN<optimizer/gradients/optimizer/sub_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
Ş
.optimizer/gradients/optimizer/sub_grad/ReshapeReshape*optimizer/gradients/optimizer/sub_grad/Sum,optimizer/gradients/optimizer/sub_grad/Shape*
Tshape0*
T0
ł
,optimizer/gradients/optimizer/sub_grad/Sum_1Sumoptimizer/gradients/AddN>optimizer/gradients/optimizer/sub_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
h
*optimizer/gradients/optimizer/sub_grad/NegNeg,optimizer/gradients/optimizer/sub_grad/Sum_1*
T0
Ž
0optimizer/gradients/optimizer/sub_grad/Reshape_1Reshape*optimizer/gradients/optimizer/sub_grad/Neg.optimizer/gradients/optimizer/sub_grad/Shape_1*
Tshape0*
T0
Ł
7optimizer/gradients/optimizer/sub_grad/tuple/group_depsNoOp/^optimizer/gradients/optimizer/sub_grad/Reshape1^optimizer/gradients/optimizer/sub_grad/Reshape_1

?optimizer/gradients/optimizer/sub_grad/tuple/control_dependencyIdentity.optimizer/gradients/optimizer/sub_grad/Reshape8^optimizer/gradients/optimizer/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/optimizer/sub_grad/Reshape

Aoptimizer/gradients/optimizer/sub_grad/tuple/control_dependency_1Identity0optimizer/gradients/optimizer/sub_grad/Reshape_18^optimizer/gradients/optimizer/sub_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimizer/gradients/optimizer/sub_grad/Reshape_1
[
*optimizer/gradients/optimizer/Q_grad/ShapeShapeoptimizer/mul*
out_type0*
T0
S
)optimizer/gradients/optimizer/Q_grad/SizeConst*
dtype0*
value	B :

(optimizer/gradients/optimizer/Q_grad/addAddoptimizer/Q/reduction_indices)optimizer/gradients/optimizer/Q_grad/Size*
T0

(optimizer/gradients/optimizer/Q_grad/modFloorMod(optimizer/gradients/optimizer/Q_grad/add)optimizer/gradients/optimizer/Q_grad/Size*
T0
U
,optimizer/gradients/optimizer/Q_grad/Shape_1Const*
dtype0*
valueB 
Z
0optimizer/gradients/optimizer/Q_grad/range/startConst*
dtype0*
value	B : 
Z
0optimizer/gradients/optimizer/Q_grad/range/deltaConst*
dtype0*
value	B :
Î
*optimizer/gradients/optimizer/Q_grad/rangeRange0optimizer/gradients/optimizer/Q_grad/range/start)optimizer/gradients/optimizer/Q_grad/Size0optimizer/gradients/optimizer/Q_grad/range/delta*

Tidx0
Y
/optimizer/gradients/optimizer/Q_grad/Fill/valueConst*
dtype0*
value	B :

)optimizer/gradients/optimizer/Q_grad/FillFill,optimizer/gradients/optimizer/Q_grad/Shape_1/optimizer/gradients/optimizer/Q_grad/Fill/value*
T0

2optimizer/gradients/optimizer/Q_grad/DynamicStitchDynamicStitch*optimizer/gradients/optimizer/Q_grad/range(optimizer/gradients/optimizer/Q_grad/mod*optimizer/gradients/optimizer/Q_grad/Shape)optimizer/gradients/optimizer/Q_grad/Fill*
T0*
N
X
.optimizer/gradients/optimizer/Q_grad/Maximum/yConst*
dtype0*
value	B :
¤
,optimizer/gradients/optimizer/Q_grad/MaximumMaximum2optimizer/gradients/optimizer/Q_grad/DynamicStitch.optimizer/gradients/optimizer/Q_grad/Maximum/y*
T0

-optimizer/gradients/optimizer/Q_grad/floordivFloorDiv*optimizer/gradients/optimizer/Q_grad/Shape,optimizer/gradients/optimizer/Q_grad/Maximum*
T0
Ĺ
,optimizer/gradients/optimizer/Q_grad/ReshapeReshapeAoptimizer/gradients/optimizer/sub_grad/tuple/control_dependency_12optimizer/gradients/optimizer/Q_grad/DynamicStitch*
Tshape0*
T0
Š
)optimizer/gradients/optimizer/Q_grad/TileTile,optimizer/gradients/optimizer/Q_grad/Reshape-optimizer/gradients/optimizer/Q_grad/floordiv*

Tmultiples0*
T0
^
,optimizer/gradients/optimizer/mul_grad/ShapeShapemain/q/BiasAdd*
out_type0*
T0
i
.optimizer/gradients/optimizer/mul_grad/Shape_1Shapeoptimizer/action_onehot*
out_type0*
T0
ź
<optimizer/gradients/optimizer/mul_grad/BroadcastGradientArgsBroadcastGradientArgs,optimizer/gradients/optimizer/mul_grad/Shape.optimizer/gradients/optimizer/mul_grad/Shape_1*
T0
~
*optimizer/gradients/optimizer/mul_grad/mulMul)optimizer/gradients/optimizer/Q_grad/Tileoptimizer/action_onehot*
T0
Á
*optimizer/gradients/optimizer/mul_grad/SumSum*optimizer/gradients/optimizer/mul_grad/mul<optimizer/gradients/optimizer/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
Ş
.optimizer/gradients/optimizer/mul_grad/ReshapeReshape*optimizer/gradients/optimizer/mul_grad/Sum,optimizer/gradients/optimizer/mul_grad/Shape*
Tshape0*
T0
w
,optimizer/gradients/optimizer/mul_grad/mul_1Mulmain/q/BiasAdd)optimizer/gradients/optimizer/Q_grad/Tile*
T0
Ç
,optimizer/gradients/optimizer/mul_grad/Sum_1Sum,optimizer/gradients/optimizer/mul_grad/mul_1>optimizer/gradients/optimizer/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
°
0optimizer/gradients/optimizer/mul_grad/Reshape_1Reshape,optimizer/gradients/optimizer/mul_grad/Sum_1.optimizer/gradients/optimizer/mul_grad/Shape_1*
Tshape0*
T0
Ł
7optimizer/gradients/optimizer/mul_grad/tuple/group_depsNoOp/^optimizer/gradients/optimizer/mul_grad/Reshape1^optimizer/gradients/optimizer/mul_grad/Reshape_1

?optimizer/gradients/optimizer/mul_grad/tuple/control_dependencyIdentity.optimizer/gradients/optimizer/mul_grad/Reshape8^optimizer/gradients/optimizer/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/optimizer/mul_grad/Reshape

Aoptimizer/gradients/optimizer/mul_grad/tuple/control_dependency_1Identity0optimizer/gradients/optimizer/mul_grad/Reshape_18^optimizer/gradients/optimizer/mul_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimizer/gradients/optimizer/mul_grad/Reshape_1
Ł
3optimizer/gradients/main/q/BiasAdd_grad/BiasAddGradBiasAddGrad?optimizer/gradients/optimizer/mul_grad/tuple/control_dependency*
T0*
data_formatNHWC
¸
8optimizer/gradients/main/q/BiasAdd_grad/tuple/group_depsNoOp@^optimizer/gradients/optimizer/mul_grad/tuple/control_dependency4^optimizer/gradients/main/q/BiasAdd_grad/BiasAddGrad

@optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependencyIdentity?optimizer/gradients/optimizer/mul_grad/tuple/control_dependency9^optimizer/gradients/main/q/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/optimizer/mul_grad/Reshape

Boptimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency_1Identity3optimizer/gradients/main/q/BiasAdd_grad/BiasAddGrad9^optimizer/gradients/main/q/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@optimizer/gradients/main/q/BiasAdd_grad/BiasAddGrad
ź
-optimizer/gradients/main/q/MatMul_grad/MatMulMatMul@optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependencymain/q/Matrix/read*
T0*
transpose_a( *
transpose_b(
¸
/optimizer/gradients/main/q/MatMul_grad/MatMul_1MatMulmain/l4/Relu@optimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
Ą
7optimizer/gradients/main/q/MatMul_grad/tuple/group_depsNoOp.^optimizer/gradients/main/q/MatMul_grad/MatMul0^optimizer/gradients/main/q/MatMul_grad/MatMul_1
˙
?optimizer/gradients/main/q/MatMul_grad/tuple/control_dependencyIdentity-optimizer/gradients/main/q/MatMul_grad/MatMul8^optimizer/gradients/main/q/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@optimizer/gradients/main/q/MatMul_grad/MatMul

Aoptimizer/gradients/main/q/MatMul_grad/tuple/control_dependency_1Identity/optimizer/gradients/main/q/MatMul_grad/MatMul_18^optimizer/gradients/main/q/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@optimizer/gradients/main/q/MatMul_grad/MatMul_1

.optimizer/gradients/main/l4/Relu_grad/ReluGradReluGrad?optimizer/gradients/main/q/MatMul_grad/tuple/control_dependencymain/l4/Relu*
T0

4optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGradBiasAddGrad.optimizer/gradients/main/l4/Relu_grad/ReluGrad*
T0*
data_formatNHWC
Š
9optimizer/gradients/main/l4/BiasAdd_grad/tuple/group_depsNoOp/^optimizer/gradients/main/l4/Relu_grad/ReluGrad5^optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGrad

Aoptimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependencyIdentity.optimizer/gradients/main/l4/Relu_grad/ReluGrad:^optimizer/gradients/main/l4/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/main/l4/Relu_grad/ReluGrad

Coptimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency_1Identity4optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGrad:^optimizer/gradients/main/l4/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@optimizer/gradients/main/l4/BiasAdd_grad/BiasAddGrad
ż
.optimizer/gradients/main/l4/MatMul_grad/MatMulMatMulAoptimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependencymain/l4/Matrix/read*
T0*
transpose_a( *
transpose_b(
ş
0optimizer/gradients/main/l4/MatMul_grad/MatMul_1MatMulmain/ReshapeAoptimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
¤
8optimizer/gradients/main/l4/MatMul_grad/tuple/group_depsNoOp/^optimizer/gradients/main/l4/MatMul_grad/MatMul1^optimizer/gradients/main/l4/MatMul_grad/MatMul_1

@optimizer/gradients/main/l4/MatMul_grad/tuple/control_dependencyIdentity.optimizer/gradients/main/l4/MatMul_grad/MatMul9^optimizer/gradients/main/l4/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/main/l4/MatMul_grad/MatMul

Boptimizer/gradients/main/l4/MatMul_grad/tuple/control_dependency_1Identity0optimizer/gradients/main/l4/MatMul_grad/MatMul_19^optimizer/gradients/main/l4/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimizer/gradients/main/l4/MatMul_grad/MatMul_1
Z
+optimizer/gradients/main/Reshape_grad/ShapeShapemain/Relu_2*
out_type0*
T0
ž
-optimizer/gradients/main/Reshape_grad/ReshapeReshape@optimizer/gradients/main/l4/MatMul_grad/tuple/control_dependency+optimizer/gradients/main/Reshape_grad/Shape*
Tshape0*
T0
~
-optimizer/gradients/main/Relu_2_grad/ReluGradReluGrad-optimizer/gradients/main/Reshape_grad/Reshapemain/Relu_2*
T0

4optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGradBiasAddGrad-optimizer/gradients/main/Relu_2_grad/ReluGrad*
T0*
data_formatNHWC
¨
9optimizer/gradients/main/l3/BiasAdd_grad/tuple/group_depsNoOp.^optimizer/gradients/main/Relu_2_grad/ReluGrad5^optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGrad

Aoptimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependencyIdentity-optimizer/gradients/main/Relu_2_grad/ReluGrad:^optimizer/gradients/main/l3/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@optimizer/gradients/main/Relu_2_grad/ReluGrad

Coptimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency_1Identity4optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGrad:^optimizer/gradients/main/l3/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@optimizer/gradients/main/l3/BiasAdd_grad/BiasAddGrad
\
-optimizer/gradients/main/l3/Conv2D_grad/ShapeShapemain/Relu_1*
out_type0*
T0
ľ
;optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-optimizer/gradients/main/l3/Conv2D_grad/Shapemain/l3/w/readAoptimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
l
/optimizer/gradients/main/l3/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"      @   @   
ś
<optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermain/Relu_1/optimizer/gradients/main/l3/Conv2D_grad/Shape_1Aoptimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
˝
8optimizer/gradients/main/l3/Conv2D_grad/tuple/group_depsNoOp<^optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInput=^optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilter

@optimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependencyIdentity;optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInput9^optimizer/gradients/main/l3/Conv2D_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropInput
Ą
Boptimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependency_1Identity<optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilter9^optimizer/gradients/main/l3/Conv2D_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimizer/gradients/main/l3/Conv2D_grad/Conv2DBackpropFilter

-optimizer/gradients/main/Relu_1_grad/ReluGradReluGrad@optimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependencymain/Relu_1*
T0

4optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGradBiasAddGrad-optimizer/gradients/main/Relu_1_grad/ReluGrad*
T0*
data_formatNHWC
¨
9optimizer/gradients/main/l2/BiasAdd_grad/tuple/group_depsNoOp.^optimizer/gradients/main/Relu_1_grad/ReluGrad5^optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGrad

Aoptimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependencyIdentity-optimizer/gradients/main/Relu_1_grad/ReluGrad:^optimizer/gradients/main/l2/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@optimizer/gradients/main/Relu_1_grad/ReluGrad

Coptimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency_1Identity4optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGrad:^optimizer/gradients/main/l2/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@optimizer/gradients/main/l2/BiasAdd_grad/BiasAddGrad
Z
-optimizer/gradients/main/l2/Conv2D_grad/ShapeShape	main/Relu*
out_type0*
T0
ľ
;optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-optimizer/gradients/main/l2/Conv2D_grad/Shapemain/l2/w/readAoptimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
l
/optimizer/gradients/main/l2/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"          @   
´
<optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter	main/Relu/optimizer/gradients/main/l2/Conv2D_grad/Shape_1Aoptimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
˝
8optimizer/gradients/main/l2/Conv2D_grad/tuple/group_depsNoOp<^optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInput=^optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilter

@optimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependencyIdentity;optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInput9^optimizer/gradients/main/l2/Conv2D_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropInput
Ą
Boptimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependency_1Identity<optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilter9^optimizer/gradients/main/l2/Conv2D_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimizer/gradients/main/l2/Conv2D_grad/Conv2DBackpropFilter

+optimizer/gradients/main/Relu_grad/ReluGradReluGrad@optimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependency	main/Relu*
T0

4optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGradBiasAddGrad+optimizer/gradients/main/Relu_grad/ReluGrad*
T0*
data_formatNHWC
Ś
9optimizer/gradients/main/l1/BiasAdd_grad/tuple/group_depsNoOp,^optimizer/gradients/main/Relu_grad/ReluGrad5^optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGrad
˙
Aoptimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependencyIdentity+optimizer/gradients/main/Relu_grad/ReluGrad:^optimizer/gradients/main/l1/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@optimizer/gradients/main/Relu_grad/ReluGrad

Coptimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency_1Identity4optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGrad:^optimizer/gradients/main/l1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@optimizer/gradients/main/l1/BiasAdd_grad/BiasAddGrad
Y
-optimizer/gradients/main/l1/Conv2D_grad/ShapeShapemain/s_t*
out_type0*
T0
ľ
;optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput-optimizer/gradients/main/l1/Conv2D_grad/Shapemain/l1/w/readAoptimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
l
/optimizer/gradients/main/l1/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"             
ł
<optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermain/s_t/optimizer/gradients/main/l1/Conv2D_grad/Shape_1Aoptimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
˝
8optimizer/gradients/main/l1/Conv2D_grad/tuple/group_depsNoOp<^optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInput=^optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilter

@optimizer/gradients/main/l1/Conv2D_grad/tuple/control_dependencyIdentity;optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInput9^optimizer/gradients/main/l1/Conv2D_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropInput
Ą
Boptimizer/gradients/main/l1/Conv2D_grad/tuple/control_dependency_1Identity<optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilter9^optimizer/gradients/main/l1/Conv2D_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimizer/gradients/main/l1/Conv2D_grad/Conv2DBackpropFilter

,optimizer/main/l1/w/RMSProp/Initializer/onesConst*
dtype0*
_class
loc:@main/l1/w*%
valueB *  ?

optimizer/main/l1/w/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_class
loc:@main/l1/w
Ç
"optimizer/main/l1/w/RMSProp/AssignAssignoptimizer/main/l1/w/RMSProp,optimizer/main/l1/w/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l1/w
p
 optimizer/main/l1/w/RMSProp/readIdentityoptimizer/main/l1/w/RMSProp*
T0*
_class
loc:@main/l1/w

/optimizer/main/l1/w/RMSProp_1/Initializer/zerosConst*
dtype0*
_class
loc:@main/l1/w*%
valueB *    

optimizer/main/l1/w/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape: *
_class
loc:@main/l1/w
Î
$optimizer/main/l1/w/RMSProp_1/AssignAssignoptimizer/main/l1/w/RMSProp_1/optimizer/main/l1/w/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l1/w
t
"optimizer/main/l1/w/RMSProp_1/readIdentityoptimizer/main/l1/w/RMSProp_1*
T0*
_class
loc:@main/l1/w

1optimizer/main/l1/biases/RMSProp/Initializer/onesConst*
dtype0*!
_class
loc:@main/l1/biases*
valueB *  ?

 optimizer/main/l1/biases/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape: *!
_class
loc:@main/l1/biases
Ű
'optimizer/main/l1/biases/RMSProp/AssignAssign optimizer/main/l1/biases/RMSProp1optimizer/main/l1/biases/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l1/biases

%optimizer/main/l1/biases/RMSProp/readIdentity optimizer/main/l1/biases/RMSProp*
T0*!
_class
loc:@main/l1/biases

4optimizer/main/l1/biases/RMSProp_1/Initializer/zerosConst*
dtype0*!
_class
loc:@main/l1/biases*
valueB *    

"optimizer/main/l1/biases/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape: *!
_class
loc:@main/l1/biases
â
)optimizer/main/l1/biases/RMSProp_1/AssignAssign"optimizer/main/l1/biases/RMSProp_14optimizer/main/l1/biases/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l1/biases

'optimizer/main/l1/biases/RMSProp_1/readIdentity"optimizer/main/l1/biases/RMSProp_1*
T0*!
_class
loc:@main/l1/biases

,optimizer/main/l2/w/RMSProp/Initializer/onesConst*
dtype0*
_class
loc:@main/l2/w*%
valueB @*  ?

optimizer/main/l2/w/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape: @*
_class
loc:@main/l2/w
Ç
"optimizer/main/l2/w/RMSProp/AssignAssignoptimizer/main/l2/w/RMSProp,optimizer/main/l2/w/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l2/w
p
 optimizer/main/l2/w/RMSProp/readIdentityoptimizer/main/l2/w/RMSProp*
T0*
_class
loc:@main/l2/w

/optimizer/main/l2/w/RMSProp_1/Initializer/zerosConst*
dtype0*
_class
loc:@main/l2/w*%
valueB @*    

optimizer/main/l2/w/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape: @*
_class
loc:@main/l2/w
Î
$optimizer/main/l2/w/RMSProp_1/AssignAssignoptimizer/main/l2/w/RMSProp_1/optimizer/main/l2/w/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l2/w
t
"optimizer/main/l2/w/RMSProp_1/readIdentityoptimizer/main/l2/w/RMSProp_1*
T0*
_class
loc:@main/l2/w

1optimizer/main/l2/biases/RMSProp/Initializer/onesConst*
dtype0*!
_class
loc:@main/l2/biases*
valueB@*  ?

 optimizer/main/l2/biases/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape:@*!
_class
loc:@main/l2/biases
Ű
'optimizer/main/l2/biases/RMSProp/AssignAssign optimizer/main/l2/biases/RMSProp1optimizer/main/l2/biases/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l2/biases

%optimizer/main/l2/biases/RMSProp/readIdentity optimizer/main/l2/biases/RMSProp*
T0*!
_class
loc:@main/l2/biases

4optimizer/main/l2/biases/RMSProp_1/Initializer/zerosConst*
dtype0*!
_class
loc:@main/l2/biases*
valueB@*    

"optimizer/main/l2/biases/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape:@*!
_class
loc:@main/l2/biases
â
)optimizer/main/l2/biases/RMSProp_1/AssignAssign"optimizer/main/l2/biases/RMSProp_14optimizer/main/l2/biases/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l2/biases

'optimizer/main/l2/biases/RMSProp_1/readIdentity"optimizer/main/l2/biases/RMSProp_1*
T0*!
_class
loc:@main/l2/biases

,optimizer/main/l3/w/RMSProp/Initializer/onesConst*
dtype0*
_class
loc:@main/l3/w*%
valueB@@*  ?

optimizer/main/l3/w/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape:@@*
_class
loc:@main/l3/w
Ç
"optimizer/main/l3/w/RMSProp/AssignAssignoptimizer/main/l3/w/RMSProp,optimizer/main/l3/w/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l3/w
p
 optimizer/main/l3/w/RMSProp/readIdentityoptimizer/main/l3/w/RMSProp*
T0*
_class
loc:@main/l3/w

/optimizer/main/l3/w/RMSProp_1/Initializer/zerosConst*
dtype0*
_class
loc:@main/l3/w*%
valueB@@*    

optimizer/main/l3/w/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape:@@*
_class
loc:@main/l3/w
Î
$optimizer/main/l3/w/RMSProp_1/AssignAssignoptimizer/main/l3/w/RMSProp_1/optimizer/main/l3/w/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l3/w
t
"optimizer/main/l3/w/RMSProp_1/readIdentityoptimizer/main/l3/w/RMSProp_1*
T0*
_class
loc:@main/l3/w

1optimizer/main/l3/biases/RMSProp/Initializer/onesConst*
dtype0*!
_class
loc:@main/l3/biases*
valueB@*  ?

 optimizer/main/l3/biases/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape:@*!
_class
loc:@main/l3/biases
Ű
'optimizer/main/l3/biases/RMSProp/AssignAssign optimizer/main/l3/biases/RMSProp1optimizer/main/l3/biases/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l3/biases

%optimizer/main/l3/biases/RMSProp/readIdentity optimizer/main/l3/biases/RMSProp*
T0*!
_class
loc:@main/l3/biases

4optimizer/main/l3/biases/RMSProp_1/Initializer/zerosConst*
dtype0*!
_class
loc:@main/l3/biases*
valueB@*    

"optimizer/main/l3/biases/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape:@*!
_class
loc:@main/l3/biases
â
)optimizer/main/l3/biases/RMSProp_1/AssignAssign"optimizer/main/l3/biases/RMSProp_14optimizer/main/l3/biases/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l3/biases

'optimizer/main/l3/biases/RMSProp_1/readIdentity"optimizer/main/l3/biases/RMSProp_1*
T0*!
_class
loc:@main/l3/biases

1optimizer/main/l4/Matrix/RMSProp/Initializer/onesConst*
dtype0*!
_class
loc:@main/l4/Matrix*
valueB
Ŕ*  ?

 optimizer/main/l4/Matrix/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape:
Ŕ*!
_class
loc:@main/l4/Matrix
Ű
'optimizer/main/l4/Matrix/RMSProp/AssignAssign optimizer/main/l4/Matrix/RMSProp1optimizer/main/l4/Matrix/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l4/Matrix

%optimizer/main/l4/Matrix/RMSProp/readIdentity optimizer/main/l4/Matrix/RMSProp*
T0*!
_class
loc:@main/l4/Matrix

4optimizer/main/l4/Matrix/RMSProp_1/Initializer/zerosConst*
dtype0*!
_class
loc:@main/l4/Matrix*
valueB
Ŕ*    

"optimizer/main/l4/Matrix/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape:
Ŕ*!
_class
loc:@main/l4/Matrix
â
)optimizer/main/l4/Matrix/RMSProp_1/AssignAssign"optimizer/main/l4/Matrix/RMSProp_14optimizer/main/l4/Matrix/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*!
_class
loc:@main/l4/Matrix

'optimizer/main/l4/Matrix/RMSProp_1/readIdentity"optimizer/main/l4/Matrix/RMSProp_1*
T0*!
_class
loc:@main/l4/Matrix

/optimizer/main/l4/bias/RMSProp/Initializer/onesConst*
dtype0*
_class
loc:@main/l4/bias*
valueB*  ?

optimizer/main/l4/bias/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_class
loc:@main/l4/bias
Ó
%optimizer/main/l4/bias/RMSProp/AssignAssignoptimizer/main/l4/bias/RMSProp/optimizer/main/l4/bias/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l4/bias
y
#optimizer/main/l4/bias/RMSProp/readIdentityoptimizer/main/l4/bias/RMSProp*
T0*
_class
loc:@main/l4/bias

2optimizer/main/l4/bias/RMSProp_1/Initializer/zerosConst*
dtype0*
_class
loc:@main/l4/bias*
valueB*    

 optimizer/main/l4/bias/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_class
loc:@main/l4/bias
Ú
'optimizer/main/l4/bias/RMSProp_1/AssignAssign optimizer/main/l4/bias/RMSProp_12optimizer/main/l4/bias/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/l4/bias
}
%optimizer/main/l4/bias/RMSProp_1/readIdentity optimizer/main/l4/bias/RMSProp_1*
T0*
_class
loc:@main/l4/bias

0optimizer/main/q/Matrix/RMSProp/Initializer/onesConst*
dtype0* 
_class
loc:@main/q/Matrix*
valueB	*  ?

optimizer/main/q/Matrix/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape:	* 
_class
loc:@main/q/Matrix
×
&optimizer/main/q/Matrix/RMSProp/AssignAssignoptimizer/main/q/Matrix/RMSProp0optimizer/main/q/Matrix/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0* 
_class
loc:@main/q/Matrix
|
$optimizer/main/q/Matrix/RMSProp/readIdentityoptimizer/main/q/Matrix/RMSProp*
T0* 
_class
loc:@main/q/Matrix

3optimizer/main/q/Matrix/RMSProp_1/Initializer/zerosConst*
dtype0* 
_class
loc:@main/q/Matrix*
valueB	*    

!optimizer/main/q/Matrix/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape:	* 
_class
loc:@main/q/Matrix
Ţ
(optimizer/main/q/Matrix/RMSProp_1/AssignAssign!optimizer/main/q/Matrix/RMSProp_13optimizer/main/q/Matrix/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0* 
_class
loc:@main/q/Matrix

&optimizer/main/q/Matrix/RMSProp_1/readIdentity!optimizer/main/q/Matrix/RMSProp_1*
T0* 
_class
loc:@main/q/Matrix

.optimizer/main/q/bias/RMSProp/Initializer/onesConst*
dtype0*
_class
loc:@main/q/bias*
valueB*  ?

optimizer/main/q/bias/RMSProp
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_class
loc:@main/q/bias
Ď
$optimizer/main/q/bias/RMSProp/AssignAssignoptimizer/main/q/bias/RMSProp.optimizer/main/q/bias/RMSProp/Initializer/ones*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/q/bias
v
"optimizer/main/q/bias/RMSProp/readIdentityoptimizer/main/q/bias/RMSProp*
T0*
_class
loc:@main/q/bias

1optimizer/main/q/bias/RMSProp_1/Initializer/zerosConst*
dtype0*
_class
loc:@main/q/bias*
valueB*    

optimizer/main/q/bias/RMSProp_1
VariableV2*
	container *
shared_name *
dtype0*
shape:*
_class
loc:@main/q/bias
Ö
&optimizer/main/q/bias/RMSProp_1/AssignAssignoptimizer/main/q/bias/RMSProp_11optimizer/main/q/bias/RMSProp_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@main/q/bias
z
$optimizer/main/q/bias/RMSProp_1/readIdentityoptimizer/main/q/bias/RMSProp_1*
T0*
_class
loc:@main/q/bias
D
optimizer/RMSProp/decayConst*
dtype0*
valueB
 *fff?
G
optimizer/RMSProp/momentumConst*
dtype0*
valueB
 *33s?
F
optimizer/RMSProp/epsilonConst*
dtype0*
valueB
 *
×#<
ç
/optimizer/RMSProp/update_main/l1/w/ApplyRMSPropApplyRMSProp	main/l1/woptimizer/main/l1/w/RMSPropoptimizer/main/l1/w/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonBoptimizer/gradients/main/l1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@main/l1/w

4optimizer/RMSProp/update_main/l1/biases/ApplyRMSPropApplyRMSPropmain/l1/biases optimizer/main/l1/biases/RMSProp"optimizer/main/l1/biases/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonCoptimizer/gradients/main/l1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@main/l1/biases
ç
/optimizer/RMSProp/update_main/l2/w/ApplyRMSPropApplyRMSProp	main/l2/woptimizer/main/l2/w/RMSPropoptimizer/main/l2/w/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonBoptimizer/gradients/main/l2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@main/l2/w

4optimizer/RMSProp/update_main/l2/biases/ApplyRMSPropApplyRMSPropmain/l2/biases optimizer/main/l2/biases/RMSProp"optimizer/main/l2/biases/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonCoptimizer/gradients/main/l2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@main/l2/biases
ç
/optimizer/RMSProp/update_main/l3/w/ApplyRMSPropApplyRMSProp	main/l3/woptimizer/main/l3/w/RMSPropoptimizer/main/l3/w/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonBoptimizer/gradients/main/l3/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@main/l3/w

4optimizer/RMSProp/update_main/l3/biases/ApplyRMSPropApplyRMSPropmain/l3/biases optimizer/main/l3/biases/RMSProp"optimizer/main/l3/biases/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonCoptimizer/gradients/main/l3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@main/l3/biases

4optimizer/RMSProp/update_main/l4/Matrix/ApplyRMSPropApplyRMSPropmain/l4/Matrix optimizer/main/l4/Matrix/RMSProp"optimizer/main/l4/Matrix/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonBoptimizer/gradients/main/l4/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@main/l4/Matrix
÷
2optimizer/RMSProp/update_main/l4/bias/ApplyRMSPropApplyRMSPropmain/l4/biasoptimizer/main/l4/bias/RMSProp optimizer/main/l4/bias/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonCoptimizer/gradients/main/l4/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@main/l4/bias
ú
3optimizer/RMSProp/update_main/q/Matrix/ApplyRMSPropApplyRMSPropmain/q/Matrixoptimizer/main/q/Matrix/RMSProp!optimizer/main/q/Matrix/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonAoptimizer/gradients/main/q/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@main/q/Matrix
ń
1optimizer/RMSProp/update_main/q/bias/ApplyRMSPropApplyRMSPropmain/q/biasoptimizer/main/q/bias/RMSPropoptimizer/main/q/bias/RMSProp_1optimizer/Maximumoptimizer/RMSProp/decayoptimizer/RMSProp/momentumoptimizer/RMSProp/epsilonBoptimizer/gradients/main/q/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@main/q/bias
Ş
optimizer/RMSPropNoOp0^optimizer/RMSProp/update_main/l1/w/ApplyRMSProp5^optimizer/RMSProp/update_main/l1/biases/ApplyRMSProp0^optimizer/RMSProp/update_main/l2/w/ApplyRMSProp5^optimizer/RMSProp/update_main/l2/biases/ApplyRMSProp0^optimizer/RMSProp/update_main/l3/w/ApplyRMSProp5^optimizer/RMSProp/update_main/l3/biases/ApplyRMSProp5^optimizer/RMSProp/update_main/l4/Matrix/ApplyRMSProp3^optimizer/RMSProp/update_main/l4/bias/ApplyRMSProp4^optimizer/RMSProp/update_main/q/Matrix/ApplyRMSProp2^optimizer/RMSProp/update_main/q/bias/ApplyRMSProp
A
summary/average.rewardPlaceholder*
shape:*
dtype0
^
summary/average.reward_1/tagsConst*
dtype0*)
value B Bsummary/average.reward_1
i
summary/average.reward_1ScalarSummarysummary/average.reward_1/tagssummary/average.reward*
T0
?
summary/average.lossPlaceholder*
shape:*
dtype0
Z
summary/average.loss_1/tagsConst*
dtype0*'
valueB Bsummary/average.loss_1
c
summary/average.loss_1ScalarSummarysummary/average.loss_1/tagssummary/average.loss*
T0
<
summary/average.qPlaceholder*
shape:*
dtype0
T
summary/average.q_1/tagsConst*
dtype0*$
valueB Bsummary/average.q_1
Z
summary/average.q_1ScalarSummarysummary/average.q_1/tagssummary/average.q*
T0
E
summary/episode.max_rewardPlaceholder*
shape:*
dtype0
f
!summary/episode.max_reward_1/tagsConst*
dtype0*-
value$B" Bsummary/episode.max_reward_1
u
summary/episode.max_reward_1ScalarSummary!summary/episode.max_reward_1/tagssummary/episode.max_reward*
T0
E
summary/episode.min_rewardPlaceholder*
shape:*
dtype0
f
!summary/episode.min_reward_1/tagsConst*
dtype0*-
value$B" Bsummary/episode.min_reward_1
u
summary/episode.min_reward_1ScalarSummary!summary/episode.min_reward_1/tagssummary/episode.min_reward*
T0
E
summary/episode.avg_rewardPlaceholder*
shape:*
dtype0
f
!summary/episode.avg_reward_1/tagsConst*
dtype0*-
value$B" Bsummary/episode.avg_reward_1
u
summary/episode.avg_reward_1ScalarSummary!summary/episode.avg_reward_1/tagssummary/episode.avg_reward*
T0
F
summary/episode.num_of_gamePlaceholder*
shape:*
dtype0
h
"summary/episode.num_of_game_1/tagsConst*
dtype0*.
value%B# Bsummary/episode.num_of_game_1
x
summary/episode.num_of_game_1ScalarSummary"summary/episode.num_of_game_1/tagssummary/episode.num_of_game*
T0
I
summary/training.learning_ratePlaceholder*
shape:*
dtype0
n
%summary/training.learning_rate_1/tagsConst*
dtype0*1
value(B& B summary/training.learning_rate_1

 summary/training.learning_rate_1ScalarSummary%summary/training.learning_rate_1/tagssummary/training.learning_rate*
T0
4
	summary/ePlaceholder*
shape:*
dtype0
D
summary/e_1/tagsConst*
dtype0*
valueB Bsummary/e_1
B
summary/e_1ScalarSummarysummary/e_1/tags	summary/e*
T0
B
summary/episode.rewardsPlaceholder*
shape:*
dtype0
_
summary/episode.rewards_1/tagConst*
dtype0**
value!B Bsummary/episode.rewards_1
n
summary/episode.rewards_1HistogramSummarysummary/episode.rewards_1/tagsummary/episode.rewards*
T0
B
summary/episode.actionsPlaceholder*
shape:*
dtype0
_
summary/episode.actions_1/tagConst*
dtype0**
value!B Bsummary/episode.actions_1
n
summary/episode.actions_1HistogramSummarysummary/episode.actions_1/tagsummary/episode.actions*
T0
á

initNoOp^step/step/Assign^main/l1/w/Assign^main/l1/biases/Assign^main/l2/w/Assign^main/l2/biases/Assign^main/l3/w/Assign^main/l3/biases/Assign^main/l4/Matrix/Assign^main/l4/bias/Assign^main/q/Matrix/Assign^main/q/bias/Assign^target/target_l1/w/Assign^target/target_l1/biases/Assign^target/target_l2/w/Assign^target/target_l2/biases/Assign^target/target_l3/w/Assign^target/target_l3/biases/Assign^target/target_l4/Matrix/Assign^target/target_l4/bias/Assign^target/target_q/Matrix/Assign^target/target_q/bias/Assign#^optimizer/main/l1/w/RMSProp/Assign%^optimizer/main/l1/w/RMSProp_1/Assign(^optimizer/main/l1/biases/RMSProp/Assign*^optimizer/main/l1/biases/RMSProp_1/Assign#^optimizer/main/l2/w/RMSProp/Assign%^optimizer/main/l2/w/RMSProp_1/Assign(^optimizer/main/l2/biases/RMSProp/Assign*^optimizer/main/l2/biases/RMSProp_1/Assign#^optimizer/main/l3/w/RMSProp/Assign%^optimizer/main/l3/w/RMSProp_1/Assign(^optimizer/main/l3/biases/RMSProp/Assign*^optimizer/main/l3/biases/RMSProp_1/Assign(^optimizer/main/l4/Matrix/RMSProp/Assign*^optimizer/main/l4/Matrix/RMSProp_1/Assign&^optimizer/main/l4/bias/RMSProp/Assign(^optimizer/main/l4/bias/RMSProp_1/Assign'^optimizer/main/q/Matrix/RMSProp/Assign)^optimizer/main/q/Matrix/RMSProp_1/Assign%^optimizer/main/q/bias/RMSProp/Assign'^optimizer/main/q/bias/RMSProp_1/Assign"