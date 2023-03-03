// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.21.7
// source: tensorflow/core/protobuf/remote_tensor_handle.proto

package for_core_protos_go_proto

import (
	tensor_shape_go_proto "github.com/galeone/tensorflow/tensorflow/go/core/framework/tensor_shape_go_proto"
	types_go_proto "github.com/galeone/tensorflow/tensorflow/go/core/framework/types_go_proto"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type ResourceDtypeAndShape struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Dtype types_go_proto.DataType                 `protobuf:"varint,1,opt,name=dtype,proto3,enum=tensorflow.DataType" json:"dtype,omitempty"`
	Shape *tensor_shape_go_proto.TensorShapeProto `protobuf:"bytes,2,opt,name=shape,proto3" json:"shape,omitempty"`
}

func (x *ResourceDtypeAndShape) Reset() {
	*x = ResourceDtypeAndShape{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ResourceDtypeAndShape) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ResourceDtypeAndShape) ProtoMessage() {}

func (x *ResourceDtypeAndShape) ProtoReflect() protoreflect.Message {
	mi := &file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ResourceDtypeAndShape.ProtoReflect.Descriptor instead.
func (*ResourceDtypeAndShape) Descriptor() ([]byte, []int) {
	return file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescGZIP(), []int{0}
}

func (x *ResourceDtypeAndShape) GetDtype() types_go_proto.DataType {
	if x != nil {
		return x.Dtype
	}
	return types_go_proto.DataType(0)
}

func (x *ResourceDtypeAndShape) GetShape() *tensor_shape_go_proto.TensorShapeProto {
	if x != nil {
		return x.Shape
	}
	return nil
}

type RemoteTensorHandle struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// The ID of the operation that produced this tensor.
	OpId int64 `protobuf:"varint,1,opt,name=op_id,json=opId,proto3" json:"op_id,omitempty"`
	// The index into the outputs of the operation that produced this tensor.
	OutputNum int32 `protobuf:"varint,2,opt,name=output_num,json=outputNum,proto3" json:"output_num,omitempty"`
	// Device where the tensor is located. Cannot be empty.
	// For multi-device functions, it's the default device passed to placer.
	Device string `protobuf:"bytes,3,opt,name=device,proto3" json:"device,omitempty"`
	// Device of the operation producing this tensor. Can be empty if the
	// operation producing this tensor is a multi-device function.
	OpDevice string `protobuf:"bytes,4,opt,name=op_device,json=opDevice,proto3" json:"op_device,omitempty"`
	// Tensor type.
	Dtype types_go_proto.DataType `protobuf:"varint,5,opt,name=dtype,proto3,enum=tensorflow.DataType" json:"dtype,omitempty"`
	// Optional data types and shapes of a remote resource variable.
	ResourceDtypesAndShapes []*ResourceDtypeAndShape `protobuf:"bytes,6,rep,name=resource_dtypes_and_shapes,json=resourceDtypesAndShapes,proto3" json:"resource_dtypes_and_shapes,omitempty"`
}

func (x *RemoteTensorHandle) Reset() {
	*x = RemoteTensorHandle{}
	if protoimpl.UnsafeEnabled {
		mi := &file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *RemoteTensorHandle) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*RemoteTensorHandle) ProtoMessage() {}

func (x *RemoteTensorHandle) ProtoReflect() protoreflect.Message {
	mi := &file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use RemoteTensorHandle.ProtoReflect.Descriptor instead.
func (*RemoteTensorHandle) Descriptor() ([]byte, []int) {
	return file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescGZIP(), []int{1}
}

func (x *RemoteTensorHandle) GetOpId() int64 {
	if x != nil {
		return x.OpId
	}
	return 0
}

func (x *RemoteTensorHandle) GetOutputNum() int32 {
	if x != nil {
		return x.OutputNum
	}
	return 0
}

func (x *RemoteTensorHandle) GetDevice() string {
	if x != nil {
		return x.Device
	}
	return ""
}

func (x *RemoteTensorHandle) GetOpDevice() string {
	if x != nil {
		return x.OpDevice
	}
	return ""
}

func (x *RemoteTensorHandle) GetDtype() types_go_proto.DataType {
	if x != nil {
		return x.Dtype
	}
	return types_go_proto.DataType(0)
}

func (x *RemoteTensorHandle) GetResourceDtypesAndShapes() []*ResourceDtypeAndShape {
	if x != nil {
		return x.ResourceDtypesAndShapes
	}
	return nil
}

var File_tensorflow_core_protobuf_remote_tensor_handle_proto protoreflect.FileDescriptor

var file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDesc = []byte{
	0x0a, 0x33, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x63, 0x6f, 0x72,
	0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x72, 0x65, 0x6d, 0x6f, 0x74,
	0x65, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x5f, 0x68, 0x61, 0x6e, 0x64, 0x6c, 0x65, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x10, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f,
	0x77, 0x2e, 0x65, 0x61, 0x67, 0x65, 0x72, 0x1a, 0x2c, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66,
	0x6c, 0x6f, 0x77, 0x2f, 0x63, 0x6f, 0x72, 0x65, 0x2f, 0x66, 0x72, 0x61, 0x6d, 0x65, 0x77, 0x6f,
	0x72, 0x6b, 0x2f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x1a, 0x25, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f,
	0x77, 0x2f, 0x63, 0x6f, 0x72, 0x65, 0x2f, 0x66, 0x72, 0x61, 0x6d, 0x65, 0x77, 0x6f, 0x72, 0x6b,
	0x2f, 0x74, 0x79, 0x70, 0x65, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22, 0x77, 0x0a, 0x15,
	0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x44, 0x74, 0x79, 0x70, 0x65, 0x41, 0x6e, 0x64,
	0x53, 0x68, 0x61, 0x70, 0x65, 0x12, 0x2a, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x0e, 0x32, 0x14, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f,
	0x77, 0x2e, 0x44, 0x61, 0x74, 0x61, 0x54, 0x79, 0x70, 0x65, 0x52, 0x05, 0x64, 0x74, 0x79, 0x70,
	0x65, 0x12, 0x32, 0x0a, 0x05, 0x73, 0x68, 0x61, 0x70, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b,
	0x32, 0x1c, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2e, 0x54, 0x65,
	0x6e, 0x73, 0x6f, 0x72, 0x53, 0x68, 0x61, 0x70, 0x65, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x52, 0x05,
	0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x8f, 0x02, 0x0a, 0x12, 0x52, 0x65, 0x6d, 0x6f, 0x74, 0x65,
	0x54, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x48, 0x61, 0x6e, 0x64, 0x6c, 0x65, 0x12, 0x13, 0x0a, 0x05,
	0x6f, 0x70, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x03, 0x52, 0x04, 0x6f, 0x70, 0x49,
	0x64, 0x12, 0x1d, 0x0a, 0x0a, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x6e, 0x75, 0x6d, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x09, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x4e, 0x75, 0x6d,
	0x12, 0x16, 0x0a, 0x06, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x06, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x12, 0x1b, 0x0a, 0x09, 0x6f, 0x70, 0x5f, 0x64,
	0x65, 0x76, 0x69, 0x63, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x09, 0x52, 0x08, 0x6f, 0x70, 0x44,
	0x65, 0x76, 0x69, 0x63, 0x65, 0x12, 0x2a, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x18, 0x05,
	0x20, 0x01, 0x28, 0x0e, 0x32, 0x14, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f,
	0x77, 0x2e, 0x44, 0x61, 0x74, 0x61, 0x54, 0x79, 0x70, 0x65, 0x52, 0x05, 0x64, 0x74, 0x79, 0x70,
	0x65, 0x12, 0x64, 0x0a, 0x1a, 0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x5f, 0x64, 0x74,
	0x79, 0x70, 0x65, 0x73, 0x5f, 0x61, 0x6e, 0x64, 0x5f, 0x73, 0x68, 0x61, 0x70, 0x65, 0x73, 0x18,
	0x06, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x27, 0x2e, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c,
	0x6f, 0x77, 0x2e, 0x65, 0x61, 0x67, 0x65, 0x72, 0x2e, 0x52, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63,
	0x65, 0x44, 0x74, 0x79, 0x70, 0x65, 0x41, 0x6e, 0x64, 0x53, 0x68, 0x61, 0x70, 0x65, 0x52, 0x17,
	0x72, 0x65, 0x73, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x44, 0x74, 0x79, 0x70, 0x65, 0x73, 0x41, 0x6e,
	0x64, 0x53, 0x68, 0x61, 0x70, 0x65, 0x73, 0x42, 0x8d, 0x01, 0x0a, 0x18, 0x6f, 0x72, 0x67, 0x2e,
	0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2e, 0x66, 0x72, 0x61, 0x6d, 0x65,
	0x77, 0x6f, 0x72, 0x6b, 0x42, 0x18, 0x52, 0x65, 0x6d, 0x6f, 0x74, 0x65, 0x54, 0x65, 0x6e, 0x73,
	0x6f, 0x72, 0x48, 0x61, 0x6e, 0x64, 0x6c, 0x65, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x50, 0x01,
	0x5a, 0x52, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x67, 0x61, 0x6c,
	0x65, 0x6f, 0x6e, 0x65, 0x2f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f,
	0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x66, 0x6c, 0x6f, 0x77, 0x2f, 0x67, 0x6f, 0x2f, 0x63, 0x6f,
	0x72, 0x65, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x75, 0x66, 0x2f, 0x66, 0x6f, 0x72, 0x5f,
	0x63, 0x6f, 0x72, 0x65, 0x5f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x73, 0x5f, 0x67, 0x6f, 0x5f, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0xf8, 0x01, 0x01, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescOnce sync.Once
	file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescData = file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDesc
)

func file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescGZIP() []byte {
	file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescOnce.Do(func() {
		file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescData = protoimpl.X.CompressGZIP(file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescData)
	})
	return file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDescData
}

var file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_tensorflow_core_protobuf_remote_tensor_handle_proto_goTypes = []interface{}{
	(*ResourceDtypeAndShape)(nil),                  // 0: tensorflow.eager.ResourceDtypeAndShape
	(*RemoteTensorHandle)(nil),                     // 1: tensorflow.eager.RemoteTensorHandle
	(types_go_proto.DataType)(0),                   // 2: tensorflow.DataType
	(*tensor_shape_go_proto.TensorShapeProto)(nil), // 3: tensorflow.TensorShapeProto
}
var file_tensorflow_core_protobuf_remote_tensor_handle_proto_depIdxs = []int32{
	2, // 0: tensorflow.eager.ResourceDtypeAndShape.dtype:type_name -> tensorflow.DataType
	3, // 1: tensorflow.eager.ResourceDtypeAndShape.shape:type_name -> tensorflow.TensorShapeProto
	2, // 2: tensorflow.eager.RemoteTensorHandle.dtype:type_name -> tensorflow.DataType
	0, // 3: tensorflow.eager.RemoteTensorHandle.resource_dtypes_and_shapes:type_name -> tensorflow.eager.ResourceDtypeAndShape
	4, // [4:4] is the sub-list for method output_type
	4, // [4:4] is the sub-list for method input_type
	4, // [4:4] is the sub-list for extension type_name
	4, // [4:4] is the sub-list for extension extendee
	0, // [0:4] is the sub-list for field type_name
}

func init() { file_tensorflow_core_protobuf_remote_tensor_handle_proto_init() }
func file_tensorflow_core_protobuf_remote_tensor_handle_proto_init() {
	if File_tensorflow_core_protobuf_remote_tensor_handle_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ResourceDtypeAndShape); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*RemoteTensorHandle); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_tensorflow_core_protobuf_remote_tensor_handle_proto_goTypes,
		DependencyIndexes: file_tensorflow_core_protobuf_remote_tensor_handle_proto_depIdxs,
		MessageInfos:      file_tensorflow_core_protobuf_remote_tensor_handle_proto_msgTypes,
	}.Build()
	File_tensorflow_core_protobuf_remote_tensor_handle_proto = out.File
	file_tensorflow_core_protobuf_remote_tensor_handle_proto_rawDesc = nil
	file_tensorflow_core_protobuf_remote_tensor_handle_proto_goTypes = nil
	file_tensorflow_core_protobuf_remote_tensor_handle_proto_depIdxs = nil
}
