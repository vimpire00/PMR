# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/szy.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fproto/szy.proto\x12\x07tenseal\x1a\x1cgoogle/protobuf/struct.proto\x1a\x1bgoogle/protobuf/empty.proto\".\n\x16PrecomputationResponse\x12\t\n\x01w\x18\x01 \x03(\x0c\x12\t\n\x01n\x18\x02 \x01(\x03\"=\n\x15PrecomputationRequest\x12\x11\n\trequest_w\x18\x01 \x01(\x08\x12\x11\n\trequest_n\x18\x02 \x01(\x08\"+\n\x12\x43omputationRequest\x12\t\n\x01u\x18\x01 \x03(\x0c\x12\n\n\x02xs\x18\x02 \x03(\x03\"!\n\x13\x43omputationResponse\x12\n\n\x02ys\x18\x01 \x03(\x03\"%\n\x18OnlineComputationRequest\x12\t\n\x01x\x18\x01 \x03(\x0c\"&\n\x19OnlineComputationResponse\x12\t\n\x01y\x18\x01 \x03(\x0c\")\n\x1bPreOnlineComputationRequest\x12\n\n\x02pk\x18\x01 \x01(\x0c\"0\n\x1cPreOnlineComputationResponse\x12\x10\n\x08querynum\x18\x01 \x01(\x03\"\"\n\x11\x46inalIndexRequest\x12\r\n\x05index\x18\x01 \x01(\x03\"\x1f\n\x10\x46inalAnsResponse\x12\x0b\n\x03\x61ns\x18\x01 \x01(\t2\xb2\x03\n\x03SZY\x12S\n\x0ePrecomputation\x12\x1e.tenseal.PrecomputationRequest\x1a\x1f.tenseal.PrecomputationResponse\"\x00\x12J\n\x0b\x43omputation\x12\x1b.tenseal.ComputationRequest\x1a\x1c.tenseal.ComputationResponse\"\x00\x12\\\n\x11OnlineComputation\x12!.tenseal.OnlineComputationRequest\x1a\".tenseal.OnlineComputationResponse\"\x00\x12\x65\n\x14PreOnlineComputation\x12$.tenseal.PreOnlineComputationRequest\x1a%.tenseal.PreOnlineComputationResponse\"\x00\x12\x45\n\nFinalIndex\x12\x1a.tenseal.FinalIndexRequest\x1a\x19.tenseal.FinalAnsResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.szy_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PRECOMPUTATIONRESPONSE._serialized_start=87
  _PRECOMPUTATIONRESPONSE._serialized_end=133
  _PRECOMPUTATIONREQUEST._serialized_start=135
  _PRECOMPUTATIONREQUEST._serialized_end=196
  _COMPUTATIONREQUEST._serialized_start=198
  _COMPUTATIONREQUEST._serialized_end=241
  _COMPUTATIONRESPONSE._serialized_start=243
  _COMPUTATIONRESPONSE._serialized_end=276
  _ONLINECOMPUTATIONREQUEST._serialized_start=278
  _ONLINECOMPUTATIONREQUEST._serialized_end=315
  _ONLINECOMPUTATIONRESPONSE._serialized_start=317
  _ONLINECOMPUTATIONRESPONSE._serialized_end=355
  _PREONLINECOMPUTATIONREQUEST._serialized_start=357
  _PREONLINECOMPUTATIONREQUEST._serialized_end=398
  _PREONLINECOMPUTATIONRESPONSE._serialized_start=400
  _PREONLINECOMPUTATIONRESPONSE._serialized_end=448
  _FINALINDEXREQUEST._serialized_start=450
  _FINALINDEXREQUEST._serialized_end=484
  _FINALANSRESPONSE._serialized_start=486
  _FINALANSRESPONSE._serialized_end=517
  _SZY._serialized_start=520
  _SZY._serialized_end=954
# @@protoc_insertion_point(module_scope)