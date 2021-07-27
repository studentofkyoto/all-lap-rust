use crate::contains::Contains;
use std::collections::HashSet;
use std::fmt;
use std::iter::repeat;

pub struct DirectedGraph {
    pub adj: Vec<Vec<usize>>,
}

impl fmt::Debug for DirectedGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DirectedGraph {{")?;
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

impl DirectedGraph {
    pub fn sort(&mut self) {
        for rs in self.adj.iter_mut() {
            rs.sort_unstable()
        }
    }

    pub fn from_adj(i2js: Vec<Vec<usize>>) -> Self {
        DirectedGraph { adj: i2js }
    }

    pub fn find_cycle(&self, is_node_startable: &impl Contains<usize>) -> Vec<usize> {
        let mut visited = HashSet::new();
        for i in 0..self.adj.len() {
            if !is_node_startable.contains_node(&i) {
                continue;
            }
            if let Some(cycle) = self._find_cycle(i, &mut vec![], &mut visited) {
                return cycle;
            }
        }
        return vec![];
    }

    fn _find_cycle(
        &self,
        i: usize,
        path: &mut Vec<usize>,
        visited: &mut HashSet<usize>,
    ) -> Option<Vec<usize>> {
        if visited.contains(&i) {
            return path
                .iter()
                .position(|x| *x == i)
                .map(|pos| path.iter().skip(pos).copied().collect());
        }
        visited.insert(i);

        if i >= self.adj.len() {
            return None;
        }
        for other in self.adj[i].iter() {
            path.push(i);
            let cycle = self._find_cycle(*other, path, visited);
            path.pop();
            if cycle.is_some() {
                return cycle;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::DirectedGraph;
    use crate::contains::Contains;

    struct DummySet {}
    impl Contains<usize> for DummySet {
        fn contains_node(&self, _: &usize) -> bool {
            true
        }
    }

    #[test]
    fn test_directed_graph() {
        let ok_starting_edge = DummySet {};
        for (adj, expected_cycle) in vec![
            (vec![], vec![]),
            (vec![vec![1], vec![2]], vec![]),
            (vec![vec![1], vec![0]], vec![1, 0]),
            (vec![vec![1], vec![0, 2]], vec![0, 1]),
            (vec![vec![1, 2], vec![0, 2]], vec![0, 1]),
            (vec![vec![1, 2], vec![0]], vec![0, 1]),
            (vec![vec![1], vec![2], vec![0]], vec![0, 1, 2]),
            (vec![vec![1], vec![2]], vec![]),
            (vec![vec![2], vec![2], vec![0]], vec![0, 2]),
            (vec![vec![2], vec![2], vec![1]], vec![1, 2]),
        ]
        .into_iter()
        {
            let dmg = DirectedGraph::from_adj(adj);
            let mut cycle = dmg.find_cycle(&ok_starting_edge);
            if expected_cycle.len() > 0 {
                assert!(cycle.contains(&expected_cycle[0]));
                let start = cycle.iter().position(|i| *i == expected_cycle[0]).unwrap();
                cycle.rotate_left(start);
            }
            assert_eq!(cycle, expected_cycle);
        }
    }
}
