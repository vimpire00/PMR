# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto import szy81_pb2 as proto_dot_szy81__pb2


class SZY81Stub(object):
    """服务定义
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Precomputation = channel.unary_unary(
                '/tenseal.SZY81/Precomputation',
                request_serializer=proto_dot_szy81__pb2.PrecomputationRequest.SerializeToString,
                response_deserializer=proto_dot_szy81__pb2.PrecomputationResponse.FromString,
                )
        self.Computation = channel.unary_unary(
                '/tenseal.SZY81/Computation',
                request_serializer=proto_dot_szy81__pb2.ComputationRequest.SerializeToString,
                response_deserializer=proto_dot_szy81__pb2.ComputationResponse.FromString,
                )
        self.FinalIndex = channel.unary_unary(
                '/tenseal.SZY81/FinalIndex',
                request_serializer=proto_dot_szy81__pb2.FinalIndexRequest.SerializeToString,
                response_deserializer=proto_dot_szy81__pb2.FinalAnsResponse.FromString,
                )


class SZY81Servicer(object):
    """服务定义
    """

    def Precomputation(self, request, context):
        """Precomputation service sends ~ench0
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Computation(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FinalIndex(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SZY81Servicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Precomputation': grpc.unary_unary_rpc_method_handler(
                    servicer.Precomputation,
                    request_deserializer=proto_dot_szy81__pb2.PrecomputationRequest.FromString,
                    response_serializer=proto_dot_szy81__pb2.PrecomputationResponse.SerializeToString,
            ),
            'Computation': grpc.unary_unary_rpc_method_handler(
                    servicer.Computation,
                    request_deserializer=proto_dot_szy81__pb2.ComputationRequest.FromString,
                    response_serializer=proto_dot_szy81__pb2.ComputationResponse.SerializeToString,
            ),
            'FinalIndex': grpc.unary_unary_rpc_method_handler(
                    servicer.FinalIndex,
                    request_deserializer=proto_dot_szy81__pb2.FinalIndexRequest.FromString,
                    response_serializer=proto_dot_szy81__pb2.FinalAnsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'tenseal.SZY81', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SZY81(object):
    """服务定义
    """

    @staticmethod
    def Precomputation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tenseal.SZY81/Precomputation',
            proto_dot_szy81__pb2.PrecomputationRequest.SerializeToString,
            proto_dot_szy81__pb2.PrecomputationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Computation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tenseal.SZY81/Computation',
            proto_dot_szy81__pb2.ComputationRequest.SerializeToString,
            proto_dot_szy81__pb2.ComputationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def FinalIndex(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tenseal.SZY81/FinalIndex',
            proto_dot_szy81__pb2.FinalIndexRequest.SerializeToString,
            proto_dot_szy81__pb2.FinalAnsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
