the structure is very bad leads to frequent bugs so here is how you should restructure, 

you have a dictionary of nodes that that constains every node 

every  graph or neural network is represented by a array  of root nodes of disconnect grpahs if there are more than one 
 but right now assume one connected graph and does not have disconnected grpah though one grpah may have multiple same starting points that merge at some point we will implement multiple disconnected  later.

now every node has parametes and info, also node name(/class_name which can be either a  block/operator_name or a custom name like maybe a group ) and tag(/variable_name) , now every forward connection to node is stored in  in_connections and out_connections 
like 
```
Tree{
    
    tree:name (always unique)
    list_of_nodes:string[]
    list_of_connections:connection[]
    inputs: [inputs_names, child_node_id@input_name ...],
    output: [output_names,child_node_id@output_name]
    params : {parm_name1 : {value:--,dtype:--,shape:----, etc ...} .. }
}

class Node{
    uniques_id: "generated for every node ensuring uniqueness. uuid? "
    name: "some_tree_name"
    tag: "tag",

    inputs: [inputs_names, child_node_id@input_name ...], // derived from Tree
    output: [output_names,child_node_id@output_name]  // derived from Tree

}

class connection{
    from : from_uinque_id
    from_output_name:"output_name"
    to : to_uinque_id
    to_input_name:"input_name"
    unique_id:"{from_uinque_id}_{output_name}_{to_uinque_id}_{input_name}"
    

}

and you have a Dictionary of trees  of Trees and a main tree that is ur current entire 
network
{
    "main":Tree(),
    "decoder" : Tree(),
    "convblock":Tree(),
    "MultiHeadAttention::Tree(),

    ...
    ...
    ...
}


we start with building a current connection graph so we have a seprate "dynamicMainTree" in which we expand the expanded group tree and substitute  the node that was expanded with it now, every connection that had the node's id in input or output so "to" or "from" gets replaced by "child_node_id" of " child_node_id@input_name/child_node_id@output_name" and the child_node@ is removed from the name of output/input so we get a new connection chart as well stored in the dynamicmaintree and rest are appended from the expanded tree to dynamicMainTree (also fix unique_id:"{from_uinque_id}_{output_name}_{to_uinque_id}_{input_name}"), now while drawing in rete start this dynamicMainTree, search name in tree dictionary if it is a tree's name and that tree has only one node in child  with name of a block  replace with the block else if it it has multiple child just show as block with params and input output:
"if it is a tree's name and that tree has only one node in child  with name of a block": this is used when we want to rename let's say a conv-block so   
every default block is add as a tree with one node and name is taken from only the tree the nodes's name does not contribute to the rete's node name 

while codegen : every tree name is a class and every tag/shared tag is class property name for the containing tree/class so that same tag share weight

also write a function to getInputOuptputParamsSignature(tree:TREE), all dangling input outputs and constants are obtained and shown. 
and this function also decides the tree named class __init__ and forward function's params
```



