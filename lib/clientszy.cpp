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
    ABYParty party(ROLE_CLIENT, "127.0.0.1", 7766);
    uint32_t num_triples = 5;

    std::vector<Sharing*>& sharings = party.GetSharings();

    // Wait for the server to finish the OT protocol and send the OT output shares
    // You need to implement network communication to receive the OT output shares from the server

    // Perform OT reconstruction to get the OT output shares
    std::vector<std::vector<Share*>>& client_output_shares = sharings[S_BOOL]->GetOutputShares(ROLE_CLIENT);
    std::vector<Share*> client_ot_outputs = client_output_shares[0];

    // Generate OT input shares
    sharings[S_BOOL]->Reset(num_triples);
    party.ExecCircuit();

    // Get the output shares from the client
    std::vector<std::vector<Share*>>& client_input_shares = sharings[S_BOOL]->GetOutputShares(ROLE_CLIENT);
    std::vector<Share*> client_ot_inputs = client_input_shares[0];

    // Send OT input shares to the server
    // You need to implement network communication to send these shares to the server

    // Evaluate the dot product circuit on the client
    sharings[S_ARITH]->Reset(num_triples);
    party.ExecCircuit();
    
    // Get the output shares from the client
    std::vector<std::vector<Share*>>& client_arith_output_shares = sharings[S_ARITH]->GetOutputShares(ROLE_CLIENT);
    std::vector<Share*> client_dot_product = client_arith_output_shares[0];
     
    // Perform secret sharing reconstruction to get the dot product result
    std::vector<int32_t> dot_product_result(num_triples, 0);
    for (uint32_t i = 0; i < num_triples; ++i) {
        dot_product_result[i] = client_dot_product[i]->GetValue();
    }
    //
}