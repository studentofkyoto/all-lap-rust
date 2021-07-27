use super::directedgraph::DirectedGraph;
use crate::types::{Collection, Cycle, Edge, Matching, Node, NodeGroup, NodeIndex};
use hopcroft_karp_rust::HopcroftKarp;
use itertools::Itertools;
use std::iter::repeat;
use std::fmt;

pub struct BipartiteGraph {
    adj: Vec<Vec<usize>>, // left -> [right]
}

impl fmt::Debug for BipartiteGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BipartiteGraph {{")?;
        for (i, js) in self.adj.iter().enumerate() {
            if js.is_empty() {continue}
            writeln!(f, "\t{:?} -> {:?}", i, js)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl BipartiteGraph {
    pub fn sort(&mut self) {
        for rs in self.adj.iter_mut() {
            rs.sort()
        }
    }
    pub fn from_adj(adj: Vec<Vec<NodeIndex>>) -> Self {
        Self { adj }
    }

    pub fn iter_edges(&'_ self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.adj
            .iter()
            .enumerate()
            .map(|(i, x)| repeat(i).zip(x.iter().copied()))
            .flatten()
    }

    fn is_empty(&self) -> bool {
        self.adj.iter().map(|xs| xs.len()).sum::<usize>() == 0
    }

    fn find_matching(&self) -> Matching {
        let mut hk = HopcroftKarp::new(self.adj.len());
        let l2r = hk.get_maximum_matching(&self.adj);
        let edges = l2r
            .iter()
            .enumerate()
            .map(|(l, r)| r.as_ref().map(|_r| Edge::new(l, *_r)))
            .flatten()
            .collect();
        Matching { edges }
    }

    fn as_directed(&self, matching: &Matching) -> DirectedGraph {
        let size = self.adj.len();
        let i2js = Self::edges2adj(
            self.adj
                .iter()
                .enumerate()
                .flat_map(|(l, rs)| rs.iter().map(move |r| Edge::new(l, *r))),
            size,
            matching,
        );
        DirectedGraph::from_adj(i2js)
    }

    fn edges2adj(
        edges: impl Iterator<Item = Edge>,
        lsize: usize,
        matching: &Matching,
    ) -> Vec<Vec<usize>> {
        let mut es = edges.peekable();
        if es.peek().is_none() {
            return vec![];
        }
        let mut i2js = vec![];

        for e in es {
            while i2js.len() < lsize + e.right + 1 {
                i2js.push(vec![]);
            }
            let (l, r) = (
                Node::new(NodeGroup::Left, e.left),
                Node::new(NodeGroup::Right, e.right),
            );
            let (s, t) = if matching.edges.contains(&e) {
                (l, r)
            } else {
                (r, l)
            };
            i2js[s.encode(lsize)].push(t.encode(lsize));
        }
        i2js
    }

    fn pop(&mut self, e: &Edge) {
        let rs = &self.adj[e.left];
        let idx = rs.iter().position(|&r| r == e.right).unwrap();
        self.adj[e.left].swap_remove(idx);
    }

    fn push(&mut self, e: Edge) {
        self.adj[e.left].push(e.right);
    }

    // Remove all the edges that are connected to the given edge.
    fn exclude_nodes(&mut self, edge: &Edge) -> Vec<Edge> {
        let mut edges_popped = vec![];
        let mut idxs = vec![];
        for (l, rs) in self.adj.iter().enumerate() {
            for (r_idx, &r) in rs.iter().enumerate() {
                if l == edge.left || r == edge.right {
                    idxs.push((l, r_idx));
                    edges_popped.push(Edge::new(l, r))
                }
            }
        }

        for &(l, r_idx) in idxs.iter().rev() {
            self.adj[l].swap_remove(r_idx);
        }
        edges_popped
    }
}

impl Matching {
    fn new(edges: Vec<Edge>) -> Matching {
        Matching { edges }
    }
    fn pop_by_node(&mut self, n: &Node) -> Option<Edge> {
        if let Some((idx, _)) = self.edges.iter().enumerate().find(|(_, e)| e.of_side(n.lr) == n.idx) {
            return Some(self.edges.swap_remove(idx));
        }
        None
    }

    fn push(&mut self, e: Edge) {
        self.edges.push(e)
    }

    fn find_edge_of_left<'a>(&'a self, left: &usize) -> Option<&'a Edge> {
        self.edges.iter().find(|e| e.left == *left)
    }
}

pub fn enum_maximum_matchings_iter(
    graph: &mut BipartiteGraph,
    allowed_start_nodes: &(impl Collection<Node> + Collection<usize>),
) -> Vec<Matching> {
    let mut ret = vec![];
    let matching = graph.find_matching();
    ret.push(matching.clone());
    let digraph = graph.as_directed(&matching);
    ret.extend(_enum_maximum_matchings_iter(
        graph,
        &matching,
        &digraph,
        allowed_start_nodes,
    ));
    ret
}

fn _enum_maximum_matchings_iter(
    graph: &mut BipartiteGraph,
    matching: &Matching,
    digraph: &DirectedGraph,
    allowed_start_nodes: &(impl Collection<Node> + Collection<usize>),
) -> Vec<Matching> {
    println!("0:\n\t{:#?}\n\t{:#?}\n\t{:#?}", graph, matching, digraph);
    let mut ret = vec![];
    // Step 1
    if graph.is_empty() {
        println!("1");
        return ret;
    }
    // Step 2
    println!("2");
    let _cycle = digraph.find_cycle(allowed_start_nodes);
    let lsize = graph.adj.len();
    if !_cycle.is_empty() {
        // Step 3
        println!("3");
        let cycle: Vec<_> = _cycle
            .into_iter()
            .map(|idx| Node::decode(&idx, &lsize))
            .collect();
        let cycle_edges: Vec<_> = cycle
            .iter()
            .chunks(2)
            .into_iter()
            .map(|c| {
                let items: Vec<_> = c.collect();
                Edge::new(items[0].idx, items[1].idx)
            })
            .collect();
        let maybe_edge = cycle_edges.iter().find(|e| {
            allowed_start_nodes.contains_node(&Node::new(NodeGroup::Left, e.left))
                && allowed_start_nodes.contains_node(&Node::new(NodeGroup::Right, e.right))
        });
        if maybe_edge.is_none() {
            println!("break");
            return ret;
        }
        let edge = *maybe_edge.unwrap();

        // Step 4
        // already done because we are not really finding the optimal edge
        println!("4");

        // Step 5
        // Construct new matching M' by flipping edges along the cycle, i.e. change the direction of all the edges in the circle
        let new_matching = matching.flip(&cycle_edges);
        println!("5: {:#?} {:#?}", matching, new_matching);
        ret.push(new_matching.clone());

        // Step 6 G-(e)
        // Recurse with the old matching M but without the edge e
        {
            let edges = graph.exclude_nodes(&edge);
            let mut digraph_minus = graph.as_directed(&matching);
            graph.sort();
            digraph_minus.sort();
            let _ret = _enum_maximum_matchings_iter(
                graph,
                matching,
                &digraph_minus,
                allowed_start_nodes,
            );
            println!("6: {:#?}", _ret.len());
            ret.extend(_ret);

            for e in edges {
                graph.push(e);
            }
        }

        // Step 7 G+(e)
        // Recurse with the new matching M' but without the edge e
        {
            graph.pop(&edge);

            let mut digraph_plus = graph.as_directed(&new_matching);
            graph.sort();
            digraph_plus.sort();
            println!("7-0: {:#?} {:#?}", edge, new_matching);
            let _ret = _enum_maximum_matchings_iter(
                graph,
                &new_matching,
                &digraph_plus,
                allowed_start_nodes,
            );
            println!("7: {:#?}", _ret.len());
            ret.extend(_ret);
            graph.push(edge);
        }
    } else {
        // Step 8
        // Find feasible path of length 2 in D(graph, matching)
        // This path has the form left1 -> right -> left2
        println!("8");
        let maybe_chain = digraph
            .adj
            .iter()
            .enumerate()
            .filter(move |(i, rs)| i < &lsize && !rs.is_empty())
            .filter_map(|(i, _)| matching.find_edge_of_left(&i))
            .flat_map(|e| {
                digraph.adj[e.right + lsize].iter().zip(
                    repeat(Node::decode(&e.left, &lsize)),
                ).zip(
                    repeat(Node::decode(&e.right, &lsize))
                )
            })
            .find(|((l2, _), _)| matching.find_edge_of_left(&l2).is_none())
            .map(|((l2, l1), r)| (l1, r, *l2));
        match maybe_chain {
            None => {
                println!("Early Return: {:#?}", digraph);
                return ret
            },
            Some((left, right, left_free)) => {
                // Construct M'
                // Exchange the direction of the path left1 -> right -> left2
                // to left1 <- right <- left2 in the new matching
                let mut new_match = matching.clone();
                let edge = Edge {
                    left: left_free,
                    right: right.idx,
                };
                println!("{:#?} {:#?} {:#?}", left.idx, edge.left, edge.right);
                new_match.pop_by_node(&left);
                new_match.push(edge);

                println!("{:#?}", &new_match);
                ret.push(new_match.clone());

                // Step 9: G+(e)
                {
                    let edges = graph.exclude_nodes(&edge);
                    let _ret = _enum_maximum_matchings_iter(
                        graph,
                        &new_match,
                        &graph.as_directed(&new_match),
                        allowed_start_nodes,
                    );
                    println!("9: {:#?}", _ret.len());
                    ret.extend(_ret);
                    for e in edges {
                        graph.push(e);
                    }
                };

                // Step 10: G-(e)
                {
                    // println!("{:#?} {:#?}", graph, edge);
                    graph.pop(&edge);
                    let _ret = _enum_maximum_matchings_iter(
                        graph,
                        matching,
                        &graph.as_directed(matching),
                        allowed_start_nodes,
                    );
                    println!("10: {:#?}", _ret.len());
                    ret.extend(_ret);
                    graph.push(edge);
                }
            }
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::{enum_maximum_matchings_iter, BipartiteGraph, Node};
    use crate::types::Collection;
    use rstest::rstest;

    struct DummySet {}
    impl Collection<usize> for DummySet {
        fn contains_node(&self, _: &usize) -> bool {
            true
        }
    }
    impl Collection<Node> for DummySet {
        fn contains_node(&self, _: &Node) -> bool {
            true
        }
    }

    #[rstest]
    #[case(1, 0)]
    #[case(1, 1)]
    #[case(2, 0)]
    #[case(2, 1)]
    #[case(2, 2)]
    #[case(3, 0)]
    #[case(3, 1)]
    #[case(3, 2)]
    #[case(3, 3)]
    #[case(4, 0)]
    #[case(4, 1)]
    #[case(4, 2)]
    #[case(4, 3)]
    #[case(5, 0)]
    #[case(5, 1)]
    #[case(5, 2)]
    #[case(5, 3)]
    fn test_completeness(#[case] n: usize, #[case] m: usize) {
        let ok_starting_edge = DummySet {};
        let adj = (0..n).map(|_| (0..m).collect()).collect();
        let mut graph = BipartiteGraph::from_adj(adj);
        let actual = enum_maximum_matchings_iter(&mut graph, &ok_starting_edge).len();
        let expect = (1..(n + 1)).rev().take(m).product();
        println!("n: {:#?} m: {:#?}", n, m);
        println!("{:#?} {:#?}", actual, expect);
        assert_eq!(actual, expect);
    }
}
