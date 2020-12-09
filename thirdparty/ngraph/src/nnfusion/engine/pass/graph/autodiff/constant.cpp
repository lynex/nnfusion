//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "backward_registry.hpp"

REGISTER_BACKWARD_TRANSLATOR(Constant).translator([](std::shared_ptr<GNode> forward_node,
                                                     const GNodeIndexVector& outputs_grad,
                                                     std::shared_ptr<nnfusion::graph::Graph> graph)
                                                      -> GNodeIndexVector {
    NNFUSION_CHECK(outputs_grad.size() == 1) << "constant have only 1 output, but "
                                             << outputs_grad.size() << " outputs_grad provided";
    // do nothing
    std::unordered_set<std::string> weights_const{
        "_model.bert.embeddings.word_embeddings.weight",
        "_model.bert.embeddings.token_type_embeddings.weight",
        "_model.bert.embeddings.position_embeddings.weight"};
    if (weights_const.find(forward_node->get_name()) != weights_const.end())
    {
        auto graph_outputs = graph->get_outputs();
        nnfusion::op::OpConfig::any myConfig;
        myConfig["learning_rate"] = 0.001;
        auto opt_op = std::make_shared<nnfusion::op::GenericOp>(
            forward_node->get_name() + "_sgd", "ApplyGradient", myConfig);
        auto opt_node =
            graph->add_node_and_edge(opt_op, {get_node_output(forward_node, 0), outputs_grad[0]});
        graph_outputs.push_back(opt_node);
        graph->set_outputs(graph_outputs);
    }
    return GNodeIndexVector{};
});