# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/szy81.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11proto/szy81.proto\x12\x07tenseal\x1a\x1cgoogle/protobuf/struct.proto\x1a\x1bgoogle/protobuf/empty.proto\"3\n\x16PrecomputationResponse\x12\n\n\x02h0\x18\x01 \x03(\x0c\x12\r\n\x05\x65nchs\x18\x02 \x03(\x03\")\n\x15PrecomputationRequest\x12\x10\n\x08requesth\x18\x01 \x01(\x08\"+\n\x12\x43omputationRequest\x12\t\n\x01u\x18\x01 \x03(\x0c\x12\n\n\x02xs\x18\x02 \x03(\x03\"!\n\x13\x43omputationResponse\x12\n\n\x02ys\x18\x01 \x03(\x03\"\"\n\x11\x46inalIndexRequest\x12\r\n\x05index\x18\x01 \x01(\x03\"\x1f\n\x10\x46inalAnsResponse\x12\x0b\n\x03\x61ns\x18\x01 \x01(\t2\xef\x01\n\x05SZY81\x12S\n\x0ePrecomputation\x12\x1e.tenseal.PrecomputationRequest\x1a\x1f.tenseal.PrecomputationResponse\"\x00\x12J\n\x0b\x43omputation\x12\x1b.tenseal.ComputationRequest\x1a\x1c.tenseal.ComputationResponse\"\x00\x12\x45\n\nFinalIndex\x12\x1a.tenseal.FinalIndexRequest\x1a\x19.tenseal.FinalAnsResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'proto.szy81_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PRECOMPUTATIONRESPONSE._serialized_start=89
  _PRECOMPUTATIONRESPONSE._serialized_end=140
  _PRECOMPUTATIONREQUEST._serialized_start=142
  _PRECOMPUTATIONREQUEST._serialized_end=183
  _COMPUTATIONREQUEST._serialized_start=185
  _COMPUTATIONREQUEST._serialized_end=228
  _COMPUTATIONRESPONSE._serialized_start=230
  _COMPUTATIONRESPONSE._serialized_end=263
  _FINALINDEXREQUEST._serialized_start=265
  _FINALINDEXREQUEST._serialized_end=299
  _FINALANSRESPONSE._serialized_start=301
  _FINALANSRESPONSE._serialized_end=332
  _SZY81._serialized_start=335
  _SZY81._serialized_end=574
# @@protoc_insertion_point(module_scope)