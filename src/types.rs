use std::collections::HashSet;
use std::hash::Hash;
use std::fmt;

pub trait Collection<T> {
    fn contains_node(&self, item: &T) -> bool
    where
        T: Hash + Eq;
}
pub type NodeIndex = usize;

#[derive(Hash, Eq, PartialEq, Debug, Copy, Clone)]
pub enum NodeGroup {
    Left,
    Right,
}

impl NodeGroup {
    // fn flip(self) -> Self {
    //     match self {
    //         Self::Left => Self::Right,
    //         Self::Right => Self::Left,
    //     }
    // }
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct Node {
    pub lr: NodeGroup,
    pub idx: NodeIndex,
}

impl Node {
    pub fn new(lr: NodeGroup, idx: NodeIndex) -> Self {
        Node { lr, idx }
    }
    pub fn encode(&self, lsize: usize) -> usize {
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

pub struct NodeSet {
    inner: HashSet<Node>,
    lsize: usize,
}

impl Collection<Node> for NodeSet {
    fn contains_node(&self, item: &Node) -> bool {
        self.inner.contains(item)
    }
}

impl Collection<usize> for NodeSet {
    fn contains_node(&self, item: &usize) -> bool {
        self.inner.contains(&Node::decode(item, &self.lsize))
    }
}

// extern crate either;
// use std::collections::HashSet;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Edge {
    pub left: NodeIndex,
    pub right: NodeIndex,
}

impl Edge {
    pub fn new(left: NodeIndex, right: NodeIndex) -> Self {
        Edge { left, right }
    }

    pub fn right_node(&self) -> Node {
        Node::new(NodeGroup::Right, self.right)
    }

    pub fn of_side(&self, lr: NodeGroup) -> NodeIndex {
        match lr {
            NodeGroup::Left => self.left,
            NodeGroup::Right => self.right,
        }
    }
}

// Convention; (left, right), ... ,(left, right);
pub type Cycle = Vec<Edge>;

#[derive(Clone)]
pub struct Matching {
    pub edges: Vec<Edge>,
}

impl fmt::Debug for Matching {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matching {{")?;
        for (i, edge) in self.edges.iter().enumerate() {
            write!(f, "{:?} -> {:?}", edge.left, edge.right)?;
            if i < self.edges.len() - 1 {
                write!(f, ", ")?;
            }
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Matching {
    pub fn flip(&self, cycle_edges: &Vec<Edge>) -> Self {
        let mut new_edges = self.edges.clone();
        for e in &mut new_edges {
            for i in 0..cycle_edges.len() {
                if e.left == cycle_edges[i].left {
                    let idx = if i == 0 {cycle_edges.len() - 1} else { i - 1 };
                    e.right = cycle_edges[idx].right;
                }
            }
        }
        Matching {edges: new_edges}
    }
}