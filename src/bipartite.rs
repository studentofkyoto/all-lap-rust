use crate::contains::Contains;
use crate::directedgraph::DirectedGraph;
use hopcroft_karp_rust::HopcroftKarp;
use itertools::Itertools;
use std::collections::HashSet;
use std::fmt;
use std::iter::repeat;
type NodeIndex = usize;

#[derive(Hash, Eq, PartialEq, Debug, Copy, Clone)]
pub enum NodeGroup {
    Left,
    Right,
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct Node {
    lr: NodeGroup,
    idx: NodeIndex,
}

impl Node {
    pub fn new(lr: NodeGroup, idx: NodeIndex) -> Self {
        Node { lr, idx }
    }
    fn encode(&self, lsize: usize) -> usize {
        match self.lr {
            NodeGroup::Left => self.idx,
            NodeGroup::Right => self.idx + lsize,
        }
    }
    pub fn decode(idx: &usize, lsize: &usize) -> Self {
        if idx < lsize {
            Self::new(NodeGroup::Left, *idx)
        } else {
            Self::new(NodeGroup::Right, idx - lsize)
        }
    }
}

#[derive(Clone)]
pub struct NodeSet {
    inner: HashSet<Node>,
    lsize: usize,
}

impl NodeSet {
    pub fn new(hashset: HashSet<Node>, lsize: usize) -> Self {
        Self {
            inner: hashset,
            lsize,
        }
    }
}

impl Contains<Node> for NodeSet {
    fn contains_node(&self, item: &Node) -> bool {
        self.inner.contains(item)
    }
}

impl Contains<usize> for NodeSet {
    fn contains_node(&self, item: &usize) -> bool {
        self.inner.contains(&Node::decode(item, &self.lsize))
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
struct Edge {
    left: NodeIndex,
    right: NodeIndex,
}

impl Edge {
    fn new(left: NodeIndex, right: NodeIndex) -> Self {
        Edge { left, right }
    }
}

#[derive(Clone)]
pub struct Matching {
    pub l2r: Vec<Option<NodeIndex>>,
}

impl fmt::Debug for Matching {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matching {{")?;
        for (i, edge) in self.edges().enumerate() {
            write!(f, "{:?} -> {:?}", edge.left, edge.right)?;
            if i < self.len() - 1 {
                write!(f, ", ")?;
            }
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Matching {
    pub fn new(l2r: Vec<Option<NodeIndex>>) -> Self {
        Self { l2r }
    }
    fn len(&self) -> usize {
        self.l2r.len()
    }
    fn contains(&self, edge: &Edge) -> bool {
        match self.l2r[edge.left] {
            None => false,
            Some(r) => r == edge.right,
        }
    }
    fn edges(&'_ self) -> impl Iterator<Item = Edge> + '_ {
        self.l2r
            .iter()
            .enumerate()
            .filter_map(|(l, r)| r.map(|_r| Edge::new(l, _r)))
    }

    fn flip(&self, cycle_edges: &[Edge]) -> Self {
        let mut l2r = self.l2r.clone();
        for i in 0..cycle_edges.len() {
            let l = cycle_edges[i].left;
            let i_1 = if i == 0 { cycle_edges.len() - 1 } else { i - 1 };
            let r = cycle_edges[i_1].right;
            l2r[l] = Some(r);
        }

        Matching::new(l2r)
    }

    fn pop_by_node(&mut self, n: &Node) -> Option<Edge> {
        match n.lr {
            NodeGroup::Left => {
                let ret = self.l2r[n.idx].map(|x| Edge::new(n.idx, x));
                self.l2r[n.idx] = None;
                ret
            }
            NodeGroup::Right => {
                if let Some((l, _)) = self.l2r.iter().enumerate().find(|(_, r)| match r {
                    None => false,
                    Some(r_idx) => *r_idx == n.idx,
                }) {
                    let ret = self.l2r[l].map(|r| Edge::new(l, r));
                    self.l2r[l] = None;
                    return ret;
                }
                None
            }
        }
    }

    fn push(&mut self, e: Edge) {
        self.l2r[e.left] = Some(e.right);
    }

    fn find_edge_of_left(&self, left: usize) -> Option<Edge> {
        self.l2r[left].map(|r| Edge::new(left, r))
    }

    pub fn iter_pairs(&'_ self) -> impl Iterator<Item = (NodeIndex, NodeIndex)> + '_ {
        self.l2r.iter().enumerate().flat_map(|(i, j)| match j {
            None => None,
            Some(_j) => (i, *_j).into(),
        })
    }
}

#[derive(Clone)]
pub struct BipartiteGraph {
    adj: Vec<Vec<usize>>, // left -> [right]
}

impl fmt::Debug for BipartiteGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "BipartiteGraph {{")?;
        for (i, js) in self.adj.iter().enumerate() {
            if js.is_empty() {
                continue;
            }
            writeln!(f, "\t{:?} -> {:?}", i, js)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl BipartiteGraph {
    fn sort(&mut self) {
        for rs in self.adj.iter_mut() {
            rs.sort_unstable()
        }
    }

    pub fn from_adj(adj: Vec<Vec<NodeIndex>>) -> Self {
        Self { adj }
    }

    fn is_empty(&self) -> bool {
        self.adj.iter().map(|xs| xs.len()).sum::<usize>() == 0
    }

    fn find_matching(&self) -> Matching {
        let mut hk = HopcroftKarp::new(self.adj.len());
        let l2r = hk.get_maximum_matching(&self.adj);
        Matching::new(l2r.to_owned())
    }

    pub fn as_directed(&self, matching: &Matching) -> DirectedGraph {
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
            let (s, t) = if matching.contains(&e) {
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

    pub fn iter_match(
        self,
        allowed_start_nodes: &'_ (impl Contains<Node> + Contains<usize>),
    ) -> impl Iterator<Item = Matching> + '_ {
        MaximumMatchingCalculator::from(self).into_iter(allowed_start_nodes)
    }
}

pub struct MaximumMatchingCalculator {
    graph: BipartiteGraph,
    callstack: Vec<Call>,
    is_first_returned: bool, // TODO: remove this thing
}

impl From<BipartiteGraph> for MaximumMatchingCalculator {
    fn from(graph: BipartiteGraph) -> Self {
        let matching = graph.find_matching();
        let digraph = graph.as_directed(&matching);
        Self::new(graph, matching, digraph)
    }
}

impl MaximumMatchingCalculator {
    pub fn new(graph: BipartiteGraph, matching: Matching, digraph: DirectedGraph) -> Self {
        let callstack = vec![Call::New(matching, digraph)];
        Self {
            graph,
            callstack,
            is_first_returned: false,
        }
    }

    pub fn next_item<T>(&mut self, allowed_start_nodes: &T) -> Option<Matching>
    where
        T: Contains<Node> + Contains<usize>,
    {
        while let Some(call) = self.callstack.pop() {
            if !self.is_first_returned {
                match call {
                    Call::New(m, d) => {
                        self.is_first_returned = true;
                        self.callstack.push(Call::New(m.clone(), d));
                        return Some(m);
                    }
                    _ => {
                        panic!("??")
                    }
                }
            }
            let ret = run(&mut self.graph, allowed_start_nodes, call);
            if ret.is_none() {
                continue;
            }
            let (response, ctxt) = ret.unwrap();
            self.callstack.push(ctxt);

            match response {
                Response::YieldFrom(m, d) => self.callstack.push(Call::New(m, d)),
                Response::Return(m) => {
                    return Some(m);
                }
            }
        }
        None
    }

    pub fn into_iter<T>(self, allowed_start_nodes: &'_ T) -> impl Iterator<Item = Matching> + '_
    where
        T: Contains<Node> + Contains<usize>,
    {
        MaximumMatchingIterator {
            inner: self,
            allowed_start_nodes,
        }
    }
}

pub struct MaximumMatchingIterator<'a, T> {
    inner: MaximumMatchingCalculator,
    allowed_start_nodes: &'a T,
}

impl<'a, T> Iterator for MaximumMatchingIterator<'a, T>
where
    T: Contains<Node> + Contains<usize>,
{
    type Item = Matching;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next_item(self.allowed_start_nodes)
    }
}

fn run<T>(
    graph: &mut BipartiteGraph,
    allowed_start_nodes: &T,
    call: Call,
) -> Option<(Response, Call)>
where
    T: Contains<Node> + Contains<usize>,
{
    match call {
        Call::New(matching, digraph) => {
            // Step 1
            if graph.is_empty() {
                return None;
            }
            // Step 2
            let _cycle = digraph.find_cycle(allowed_start_nodes);
            let lsize = graph.adj.len();
            if !_cycle.is_empty() {
                // Step 3
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
                let edge = cycle_edges
                    .iter()
                    .find(|e| {
                        allowed_start_nodes.contains_node(&Node::new(NodeGroup::Left, e.left))
                            && allowed_start_nodes
                                .contains_node(&Node::new(NodeGroup::Right, e.right))
                    })?
                    .to_owned();

                // Step 4 . skip

                // Step 5 flip along cycle from given matching
                let new_matching = matching.flip(&cycle_edges);
                Some((
                    Response::Return(new_matching.clone()),
                    Call::AfterStep5(new_matching, matching, edge),
                ))
            } else {
                // Step 8. feasible path of length 2: left1 -> right -> left2
                let maybe_chain = digraph
                    .adj
                    .iter()
                    .enumerate()
                    .filter(move |(i, rs)| i < &lsize && !rs.is_empty())
                    .filter_map(|(i, _)| matching.find_edge_of_left(i))
                    .flat_map(|e| {
                        digraph.adj[e.right + lsize]
                            .iter()
                            .zip(repeat(Node::decode(&e.left, &lsize)))
                            .zip(repeat(Node::decode(&e.right, &lsize)))
                    })
                    .find(|((l2, _), _)| matching.find_edge_of_left(**l2).is_none())
                    .map(|((l2, l1), r)| (l1, r, *l2));
                match maybe_chain {
                    None => None,
                    Some((left, right, left_free)) => {
                        // (left1 -> right -> left2) => (left2 -> right -> left1)
                        let mut new_match = matching.clone();
                        let edge = Edge {
                            left: left_free,
                            right: right.idx,
                        };
                        new_match.pop_by_node(&left);
                        new_match.push(edge);
                        Some((
                            Response::Return(new_match.clone()),
                            Call::AfterStep8(new_match, matching, edge),
                        ))
                    }
                }
            }
        }
        Call::AfterStep5(new_matching, matching, edge) => {
            // Step 6 G-(e)
            let edges = graph.exclude_nodes(&edge);
            let mut digraph_minus = graph.as_directed(&matching);
            graph.sort();
            digraph_minus.sort();
            Some((
                Response::YieldFrom(matching, digraph_minus),
                Call::AfterStep6(new_matching, edge, edges),
            ))
        }
        Call::AfterStep6(new_matching, edge, edges) => {
            // Wrap up previous step
            for e in edges {
                graph.push(e);
            }

            // Step 7 G+(e)
            graph.pop(&edge);

            let mut digraph_plus = graph.as_directed(&new_matching);
            graph.sort();
            digraph_plus.sort();
            Some((
                Response::YieldFrom(new_matching, digraph_plus),
                Call::AfterStep7(edge),
            ))
        }
        Call::AfterStep7(edge) => {
            // Wrap up previous step
            graph.push(edge);
            None
        }
        Call::AfterStep8(new_match, matching, edge) => {
            // Step 9: G+(e)
            let edges = graph.exclude_nodes(&edge);
            let digraph = graph.as_directed(&new_match);
            Some((
                Response::YieldFrom(new_match, digraph),
                Call::AfterStep9(matching, edge, edges),
            ))
        }
        Call::AfterStep9(matching, edge, edges) => {
            // wrap up previous step
            for e in edges {
                graph.push(e);
            }
            // Step 10: G-(e)
            graph.pop(&edge);
            let digraph = graph.as_directed(&matching);
            Some((
                Response::YieldFrom(matching, digraph),
                Call::AfterStep10(edge),
            ))
        }
        Call::AfterStep10(edge) => {
            graph.push(edge);
            None
        }
    }
}

#[derive(Debug)]
enum Call {
    New(Matching, DirectedGraph),
    AfterStep5(Matching, Matching, Edge),
    AfterStep6(Matching, Edge, Vec<Edge>),
    AfterStep7(Edge),
    AfterStep8(Matching, Matching, Edge),
    AfterStep9(Matching, Edge, Vec<Edge>),
    AfterStep10(Edge),
}

enum Response {
    YieldFrom(Matching, DirectedGraph),
    Return(Matching),
}

#[cfg(test)]
mod tests {
    use super::{BipartiteGraph, Node, NodeGroup};
    use crate::contains::Contains;
    use rstest::rstest;
    use std::collections::HashSet;
    use std::iter::FromIterator;

    struct DummySet {}
    impl Contains<usize> for DummySet {
        fn contains_node(&self, _: &usize) -> bool {
            true
        }
    }
    impl Contains<Node> for DummySet {
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
        let graph = BipartiteGraph::from_adj(adj);

        let actual = graph
            .iter_match(&ok_starting_edge)
            .collect::<Vec<_>>()
            .len();
        let expect = (1..(n + 1)).rev().take(m).product();
        println!("n: {:#?} m: {:#?}", n, m);
        println!("{:#?} {:#?}", actual, expect);
        assert_eq!(actual, expect);
    }

    struct MyHashSet {
        inner: HashSet<Node>,
        lsize: usize,
    }
    impl MyHashSet {
        fn from_nodes(nodes: impl Iterator<Item = Node>, lsize: usize) -> Self {
            Self {
                inner: HashSet::from_iter(nodes),
                lsize,
            }
        }
    }

    impl Contains<usize> for MyHashSet {
        fn contains_node(&self, i: &usize) -> bool {
            self.inner.contains(&Node::decode(i, &self.lsize))
        }
    }

    impl Contains<Node> for MyHashSet {
        fn contains_node(&self, n: &Node) -> bool {
            self.inner.contains(n)
        }
    }

    #[test]
    fn test_no_loop() {
        let adj = vec![vec![0, 2], vec![0, 1], vec![2, 3], vec![2, 3]];
        let hashset = MyHashSet::from_nodes(
            vec![
                Node::new(NodeGroup::Left, 0),
                Node::new(NodeGroup::Left, 1),
                Node::new(NodeGroup::Right, 0),
                Node::new(NodeGroup::Right, 1),
            ]
            .into_iter(),
            4,
        );

        let graph = BipartiteGraph::from_adj(adj);
        let solutions = graph.iter_match(&hashset).collect::<Vec<_>>();
        assert_eq!(solutions.len(), 1);
    }
}
