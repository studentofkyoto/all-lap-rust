use std::hash::Hash;

pub trait Contains<T> {
    fn contains_node(&self, item: &T) -> bool
    where
        T: Hash + Eq;
}
