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
// 1323: https://leetcode.cn/problems/maximum-69-number/
pub fn maximum69_number(num: i32) -> i32 {
    let mut res = 0;
    let mut state = 1;
    while state < num {
        if (num / state) % 10 == 6 {
            res = state;
        }
        state *= 10;
    }
    res * 3 + num
}
// 1403: https://leetcode.cn/problems/minimum-subsequence-in-non-increasing-order/
pub fn min_subsequence(mut nums: Vec<i32>) -> Vec<i32> {
    nums.sort();
    let sum = nums.iter().sum::<i32>() / 2;
    let mut res = Vec::with_capacity(nums.len());
    let mut state = 0;
    for i in (0..nums.len()).rev() {
        res.push(nums[i]);
        state += nums[i];
        if state > sum {
            return res;
        }
    }
    res
}
// 1710: https://leetcode.cn/problems/maximum-units-on-a-truck/
pub fn maximum_units(mut box_types: Vec<Vec<i32>>, mut truck_size: i32) -> i32 {
    box_types.sort_by(|a, b| b[1].partial_cmp(&a[1]).unwrap());
    let mut res = 0;
    let mut i = 0;
    while truck_size > 0 && i < box_types.len() {
        res += box_types[i][1];
        box_types[i][0] -= 1;
        truck_size -= 1;
        if box_types[i][0] == 0 {
            i += 1;
        }
    }
    res
}
// 1736: https://leetcode.cn/problems/latest-time-by-replacing-hidden-digits/
pub fn maximum_time(time: String) -> String {
    let mut s = Vec::with_capacity(5);
    for c in time.bytes() {
        s.push(c);
    }
    if s[0] == b'?' {
        if s[1] - b'0' < 4 || s[1] == b'?' {
            s[0] = b'2';
        } else {
            s[0] = b'1';
        }
    }
    if s[1] == b'?' {
        match s[0] {
            b'0' | b'1' => s[1] = b'9',
            _ => s[1] = b'3',
        }
    }
    if s[3] == b'?' {
        s[3] = b'5';
    }
    if s[4] == b'?' {
        s[4] = b'9';
    }

    String::from_utf8(s).unwrap()
}
// 1827: https://leetcode.cn/problems/minimum-operations-to-make-the-array-increasing/
pub fn min_operations(mut nums: Vec<i32>) -> i32 {
    let mut res = 0;
    for i in 1..nums.len() {
        let increment = nums[i - 1] - nums[i] + 1;
        if increment > 0 {
            res += increment;
            nums[i] = nums[i - 1] + 1;
        }
    }
    res
}
// 1903: https://leetcode.cn/problems/largest-odd-number-in-string/
pub fn largest_odd_number(num: String) -> String {
    let bytes = num.as_bytes();
    for i in (0..bytes.len()).rev() {
        if (bytes[i] - b'0') % 2 != 0 {
            return num[0..i + 1].to_string();
        }
    }
    String::from("")
}
// 1974: https://leetcode.cn/problems/minimum-time-to-type-word-using-special-typewriter/
pub fn min_time_to_type(word: String) -> i32 {
    let mut prev = b'a' as i32;
    let mut res = 0;
    for &c in word.as_bytes() {
        let c = c as i32;
        let gap = (c - prev).abs();
        let gap = gap.min(26 - gap);
        res += gap + 1;
        prev = c;
    }
    res
}
// 2027: https://leetcode.cn/problems/minimum-moves-to-convert-string/
pub fn minimum_moves(s: String) -> i32 {
    let s = s.as_bytes();
    let mut i = 0;
    let mut res = 0;
    while i < s.len() {
        if s[i] == b'X' {
            res += 1;
            i += 3;
        } else {
            i += 1;
        }
    }
    res
}
// 2078: https://leetcode.cn/problems/two-furthest-houses-with-different-colors/
pub fn max_distance(colors: Vec<i32>) -> i32 {
    let len = colors.len();
    let mut distance = 0;
    let mut i = 0;
    while i < len && len - i > distance {
        let mut j = len - 1;
        while j > i && colors[i] == colors[j] {
            j -= 1;
        }
        if j > i {
            distance = distance.max(j - i);
        }
        i += 1;
    }
    distance as i32
}
// 2144: https://leetcode.cn/problems/minimum-cost-of-buying-candies-with-discount/
pub fn minimum_cost(mut cost: Vec<i32>) -> i32 {
    cost.sort_by(|a, b| b.partial_cmp(&a).unwrap());
    cost.iter()
        .enumerate()
        .fold(0, |res, (idx, x)| if idx % 3 != 2 { res + x } else { res })
}
// 2160: https://leetcode.cn/problems/minimum-sum-of-four-digit-number-after-splitting-digits/
pub fn minimum_sum(num: i32) -> i32 {
    let mut digits = Vec::with_capacity(4);

    let mut state = 1;
    while state < 10000 {
        digits.push(num / state % 10);
        state *= 10;
    }
    digits.sort();

    let (mut i, mut j) = (0, 0);
    for d in digits {
        if i < j {
            i = i * 10 + d;
        } else {
            j = j * 10 + d;
        }
    }
    i + j
}
// 2224: https://leetcode.cn/problems/minimum-number-of-operations-to-convert-time/
pub fn convert_time(cur: String, cor: String) -> i32 {
    fn time_to_minutes(time: &[u8]) -> i32 {
        let h = ((time[0] - b'0') * 10 + (time[1] - b'0')) as i32;
        let m = ((time[3] - b'0') * 10 + (time[4] - b'0')) as i32;
        h * 60 + m
    }
    let mut res = 0;
    let mut gap = time_to_minutes(cor.as_bytes()) - time_to_minutes(cur.as_bytes());
    if gap >= 60 {
        res += gap / 60;
        gap %= 60;
    }
    if gap >= 15 {
        res += gap / 15;
        gap %= 15;
    }
    if gap >= 5 {
        res += gap / 5;
        gap %= 5;
    }
    res + gap
}
// 2259: https://leetcode.cn/problems/remove-digit-from-number-to-maximize-result/
pub fn remove_digit(mut number: String, digit: char) -> String {
    let digits = number.as_bytes();
    let digit = digit as u8;

    let mut i = 0;
    let mut idx = i;
    while i < digits.len() {
        if digits[i] == digit {
            idx = i;
            if i + 1 < digits.len() && digits[i + 1] > digit || i + 1 == digits.len() {
                break;
            }
        }
        i += 1;
    }
    number.remove(idx);
    number
}
// 2335: https://leetcode.cn/problems/minimum-amount-of-time-to-fill-cups/
pub fn fill_cups(mut a: Vec<i32>) -> i32 {
    a.sort();
    if a[2] > a[0] + a[1] {
        a[2]
    } else {
        (a[0] + a[1] + a[2] + 1) / 2
    }
}
// 2383: https://leetcode.cn/problems/minimum-hours-of-training-to-win-a-competition/
pub fn min_number_of_hours(mut ienrg: i32, mut iexp: i32, enrg: Vec<i32>, exp: Vec<i32>) -> i32 {
    let mut res = 0;
    for i in 0..exp.len() {
        let need_eng = enrg[i] - ienrg + 1;
        let need_exp = exp[i] - iexp + 1;
        if need_eng > 0 {
            res += need_eng;
            ienrg += need_eng;
        }
        if need_exp > 0 {
            res += need_exp;
            iexp += need_exp;
        }
        ienrg -= enrg[i];
        iexp += exp[i];
    }
    res
}
// 2389: https://leetcode.cn/problems/longest-subsequence-with-limited-sum/
pub fn answer_queries(mut nums: Vec<i32>, queries: Vec<i32>) -> Vec<i32> {
    nums.sort();
    for i in 1..nums.len() {
        nums[i] += nums[i - 1];
    }
    queries
        .iter()
        .map(|&x| {
            let (mut i, mut j) = (0, nums.len());
            while i < j - 1 {
                let mid = i + (j - i) / 2;
                if nums[mid] <= x {
                    i = mid;
                } else {
                    j = mid;
                }
            }
            if nums[i] <= x {
                i as i32 + 1
            } else {
                0
            }
        })
        .collect()
}
