#include <iostream>
#include <vector>
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <abycore/aby/abyparty.h>
#include <abycore/circuit/booleancircuits.h>
#include <abycore/circuit/arithmeticcircuits.h>
#include <abycore/sharing/sharing.h>
#include <abycore/aby/utils/stopwatch.h>

using namespace ENCRYPTO;

int main(int argc, char** argv) {
    ABYParty party(ROLE_SERVER, "127.0.0.1", 7766);
    uint32_t num_triples = 5;

    std::vector<Sharing*>& sharings = party.GetSharings();

    // Generate OT multiplication triples
    sharings[S_BOOL]->Reset(num_triples);
    party.ExecCircuit();

    // Get the output shares from the server
    std::vector<Sharing*>& server_sharings = party.GetSharings(ROLE_SERVER);
    std::vector<std::vector<Share*>>& server_output_shares = sharings[S_BOOL]->GetOutputShares(ROLE_SERVER);
    std::vector<Share*> server_ot_outputs = server_output_shares[0];

    // Send OT output shares to the client
    // You need to implement network communication to send these shares to the client

    // Wait for the client to finish the OT protocol and send the OT input shares
    // You need to implement network communication to receive the OT input shares from the client

    // Evaluate the dot product circuit on the server
    sharings[S_ARITH]->Reset(num_triples);
    party.ExecCircuit();

    // Get the output shares from the server
    std::vector<std::vector<Share*>>& server_arith_output_shares = sharings[S_ARITH]->GetOutputShares(ROLE_SERVER);
    std::vector<Share*> server_dot_product = server_arith_output_shares[0];

    // Perform secret sharing reconstruction to get the dot product result
    std::vector<int32_t> dot_product_result(num_triples, 0);
    for (uint32_t i = 0; i < num_triples; ++i) {
        dot_product_result[i] = server_dot_product[i]->GetValue();
    }

    // Output the dot product result
    for (uint32_t i = 0; i < num_triples; ++i) {
        std::cout << "Dot Product Result " << i << ": " << dot_product_result[i] << std::endl;
    }

    return 0;
}
