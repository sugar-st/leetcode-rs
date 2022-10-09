#![allow(dead_code)]
#![allow(unused_variables)]

// 1002: https://leetcode.cn/problems/find-common-characters/
pub fn common_chars(words: Vec<String>) -> Vec<String> {
    fn frequency(word: &String) -> [i8; 26] {
        word.as_bytes().iter().fold([0; 26], |mut freq, &b| {
            freq[b as usize - 'a' as usize] += 1;
            freq
        })
    }
    fn intersection(mut x: [i8; 26], y: [i8; 26]) -> [i8; 26] {
        for i in 0..26 {
            x[i] = x[i].min(y[i]);
        }
        x
    }
    fn form(x: [i8; 26]) -> Vec<String> {
        x.into_iter()
            .enumerate()
            .fold(Vec::new(), |mut res, (idx, count)| {
                for i in 0..count {
                    res.push(String::from_utf8(vec![b'a' + idx as u8]).unwrap());
                }
                res
            })
    }
    let mut freq = frequency(&words[0]);
    for i in 1..words.len() {
        freq = intersection(freq, frequency(&words[i]));
    }
    form(freq)
}
// 1005: https://leetcode.cn/problems/maximize-sum-of-array-after-k-negations/
pub fn largest_sum_after_k_negations(nums: Vec<i32>, k: i32) -> i32 {
    let mut nums = nums;
    let mut k = k as usize;
    nums.sort();
    let mut i = 0;
    while k != 0 && i < nums.len() && nums[i] < 0 {
        nums[i] = -nums[i];
        k -= 1;
        i += 1;
    }
    if k != 0 {
        if i >= nums.len() {
            i = nums.len() - 1;
        } else if i > 0 && nums[i] > nums[i - 1] {
            i = i - 1;
        }
        if k % 2 != 0 {
            nums[i] = -nums[i];
        }
    }
    nums.iter().sum()
}
// 1013: https://leetcode.cn/problems/partition-array-into-three-parts-with-equal-sum/
pub fn can_three_parts_equal_sum(mut nums: Vec<i32>) -> bool {
    for i in 1..nums.len() {
        nums[i] = nums[i - 1] + nums[i];
    }
    let mut sum = *nums.last().unwrap();
    if sum % 3 != 0 {
        return false;
    }
    sum /= 3;
    let mut count = 0;
    for i in 0..nums.len() {
        if nums[i] == sum {
            sum <<= 1;
            count += 1;
            if count == 2 && i != nums.len() - 1 {
                return true;
            }
        }
    }
    false
}
// 1018: https://leetcode.cn/problems/binary-prefix-divisible-by-5/
pub fn prefixes_div_by5(nums: Vec<i32>) -> Vec<bool> {
    // simulation, use mod to avoid overflow
    nums.iter()
        .fold(
            (Vec::with_capacity(nums.len()), 0),
            |(mut res, mut num), &x| {
                num = (num * 2 + x) % 5;
                res.push(num == 0);
                (res, num)
            },
        )
        .0
}
// 1030: https://leetcode.cn/problems/matrix-cells-in-distance-order/
pub fn all_cells_dist_order(rows: i32, cols: i32, r_center: i32, c_center: i32) -> Vec<Vec<i32>> {
    let mut res = Vec::with_capacity((rows * cols) as usize);
    res.push(vec![r_center, c_center]);
    let mut distance = 1;
    let mut count = 1;
    'o: loop {
        for i in 0..distance {
            let j = distance - i;
            for (x, y) in vec![
                (r_center + i, c_center + j),
                (r_center + j, c_center - i),
                (r_center - i, c_center - j),
                (r_center - j, c_center + i),
            ] {
                if x >= 0 && y >= 0 && x < rows && y < cols {
                    res.push(vec![x, y]);
                    count += 1;
                    if count == res.capacity() {
                        break 'o;
                    }
                }
            }
        }
        distance += 1;
    }
    res
}
// 1037: https://leetcode.cn/problems/valid-boomerang/
pub fn is_boomerang(p: Vec<Vec<i32>>) -> bool {
    // math: vector(ab) x vector(bc) != 0
    (p[0][0] - p[1][0]) * (p[0][1] - p[2][1]) != (p[0][0] - p[2][0]) * (p[0][1] - p[1][1])
}
// 1046: https://leetcode.cn/problems/last-stone-weight/
pub fn last_stone_weight(stones: Vec<i32>) -> i32 {
    // using heap for simulation
    use std::collections::BinaryHeap;
    let mut heap = stones.into_iter().collect::<BinaryHeap<i32>>();
    while heap.len() > 1 {
        let i = heap.pop().unwrap();
        let j = heap.pop().unwrap();
        if i != j {
            heap.push(i - j);
        }
    }
    if heap.len() == 1 {
        *heap.peek().unwrap()
    } else {
        0
    }
}
// 1051: https://leetcode.cn/problems/height-checker/
pub fn height_checker(heights: Vec<i32>) -> i32 {
    // sort and compare
    let mut sorted = heights.clone();
    sorted.sort();
    sorted.into_iter().zip(heights.into_iter()).fold(
        0,
        |res, (new, old)| {
            if new != old {
                res + 1
            } else {
                res
            }
        },
    )
}
// 1089: https://leetcode.cn/problems/duplicate-zeros/
pub fn duplicate_zeros(nums: &mut Vec<i32>) {
    // simulation started from behind
    let mut cnt = nums
        .iter()
        .fold(0, |cnt, x| if *x == 0 { cnt + 1 } else { cnt });
    for i in (0..nums.len()).rev() {
        if nums[i] == 0 {
            cnt -= 1;
            if i + cnt + 1 < nums.len() {
                nums[i + cnt + 1] = 0;
            }
        }
        if i + cnt < nums.len() {
            nums[i + cnt] = nums[i];
        }
    }
}
// 1122: https://leetcode.cn/problems/relative-sort-array/
pub fn relative_sort_array(arr1: Vec<i32>, arr2: Vec<i32>) -> Vec<i32> {
    use std::collections::HashMap;
    let m = arr2
        .into_iter()
        .enumerate()
        .fold(HashMap::new(), |mut m, (idx, arr)| {
            m.insert(arr, idx);
            m
        });
    let mut arr1 = arr1;
    arr1.sort_by(|a, b| {
        use std::cmp::Ordering;
        match (m.get(a), m.get(b)) {
            (Some(i), Some(j)) => i.partial_cmp(j).unwrap(),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.partial_cmp(b).unwrap(),
        }
    });
    arr1
}
// 1128: https://leetcode.cn/problems/number-of-equivalent-domino-pairs/
pub fn num_equiv_domino_pairs(dominoes: Vec<Vec<i32>>) -> i32 {
    use std::collections::HashMap;
    let m = dominoes.into_iter().fold(HashMap::new(), |mut m, d| {
        if d[0] > d[1] {
            *m.entry((d[1], d[0])).or_insert(0) += 1;
        } else {
            *m.entry((d[0], d[1])).or_insert(0) += 1;
        }
        m
    });
    m.into_iter().fold(0, |res, (_, v)| res + v * (v - 1)) / 2
}
// 1160: https://leetcode.cn/problems/find-words-that-can-be-formed-by-characters/
pub fn count_characters(words: Vec<String>, chars: String) -> i32 {
    let chars = chars.as_bytes().into_iter().fold([0; 26], |mut freq, b| {
        freq[(b - b'a') as usize] += 1;
        freq
    });
    words.into_iter().fold(0, |res, word| {
        let cnt = word.len() as i32;
        // instead of counting word, clone can reduce time consume
        let mut freq = chars.clone();
        for &b in word.as_bytes() {
            let i = (b - b'a') as usize;
            freq[i] -= 1;
            if freq[i] < 0 {
                return res;
            }
        }
        res + cnt
    })
}
// 1184: https://leetcode.cn/problems/distance-between-bus-stops/
pub fn distance_between_bus_stops(distance: Vec<i32>, start: i32, destination: i32) -> i32 {
    let min = start.min(destination) as usize;
    let max = start.max(destination) as usize;
    distance[min..max].iter().sum::<i32>().min(
        distance[0..min].iter().sum::<i32>() + distance[max..distance.len()].iter().sum::<i32>(),
    )
}
// 1200: https://leetcode.cn/problems/minimum-absolute-difference/
pub fn minimum_abs_difference(mut arr: Vec<i32>) -> Vec<Vec<i32>> {
    // sort and compare
    arr.sort();
    let mut distances = vec![0; arr.len() - 1];
    let mut res = Vec::new();
    for i in 0..arr.len() - 1 {
        distances[i] = arr[i + 1] - arr[i];
    }
    let min = *distances.iter().min().unwrap();
    for i in 0..distances.len() {
        if distances[i] == min {
            res.push(vec![arr[i], arr[i + 1]]);
        }
    }
    res
}
// 1207: https://leetcode.cn/problems/unique-number-of-occurrences/
pub fn unique_occurrences(arr: Vec<i32>) -> bool {
    // check if the length of the set of frequency is the same as the length of frequency itself
    use std::collections::{HashMap, HashSet};
    let set = arr.into_iter().fold(HashMap::new(), |mut s, x| {
        *s.entry(x).or_insert(1) += 1;
        s
    });
    set.len() == set.into_values().collect::<HashSet<i32>>().len()
}
// 1217: https://leetcode.cn/problems/minimum-cost-to-move-chips-to-the-same-position/
pub fn min_cost_to_move_chips(position: Vec<i32>) -> i32 {
    let (even, odd) =
        position.into_iter().fold(
            (0, 0),
            |(e, o), x| {
                if x % 2 == 0 {
                    (e + 1, o)
                } else {
                    (e, o + 1)
                }
            },
        );
    even.min(odd)
}
// 1221: https://leetcode.cn/problems/split-a-string-in-balanced-strings/
pub fn balanced_string_split(s: String) -> i32 {
    let (mut l, mut r) = (0, 0);
    let mut res = 0;
    let s = s.as_bytes();
    for i in 0..s.len() {
        if s[i] == b'R' {
            r += 1;
        } else {
            l += 1;
        }
        if r == l {
            res += 1;
        }
    }
    res
}
// 1232: https://leetcode.cn/problems/check-if-it-is-a-straight-line/
pub fn check_straight_line(c: Vec<Vec<i32>>) -> bool {
    for i in 1..c.len() - 1 {
        if (c[i][0] - c[i - 1][0]) * (c[i + 1][1] - c[i][1])
            != (c[i + 1][0] - c[i][0]) * (c[i][1] - c[i - 1][1])
        {
            return false;
        }
    }
    true
}
// 1252: https://leetcode.cn/problems/cells-with-odd-values-in-a-matrix/
pub fn odd_cells(m: i32, n: i32, ops: Vec<Vec<i32>>) -> i32 {
    // record the operation count for each single row and single column
    let mut row = vec![0; m as usize];
    let mut col = vec![0; n as usize];
    for op in ops {
        row[op[0] as usize] += 1;
        col[op[1] as usize] += 1;
    }
    let (odd, even) =
        col.into_iter().fold(
            (0, 0),
            |(o, e), x| {
                if x % 2 == 0 {
                    (o, e + 1)
                } else {
                    (o + 1, e)
                }
            },
        );
    row.into_iter()
        .fold(0, |res, x| if x % 2 == 0 { res + odd } else { res + even })
}
// 1260: https://leetcode.cn/problems/shift-2d-grid/
pub fn shift_grid(mut grid: Vec<Vec<i32>>, k: i32) -> Vec<Vec<i32>> {
    let (row, col) = (grid.len(), grid[0].len());
    let cnt = row * col;
    let k = k as usize % cnt;
    let index = |x: usize| -> (usize, usize) { (x / col, x % col) };
    let mut tmp = vec![0; k];
    for i in cnt - k..cnt {
        let (x, y) = index(i);
        tmp[i - cnt + k] = grid[x][y];
    }
    for i in (k..cnt).rev() {
        let (x, y) = index((i - k) % cnt);
        let (nx, ny) = index(i);
        grid[nx][ny] = grid[x][y];
    }
    for i in 0..k {
        let (x, y) = index(i);
        grid[x][y] = tmp[i];
    }
    grid
}
