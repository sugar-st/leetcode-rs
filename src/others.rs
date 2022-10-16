#![allow(dead_code)]

// LCP 33: https://leetcode.cn/problems/o8SXZn/
pub fn store_water(bucket: Vec<i32>, vat: Vec<i32>) -> i32 {
    let n = vat.len();
    let &max_opt = vat.iter().max().unwrap();

    if max_opt == 0 {
        return 0;
    }

    let mut ans = 10000;

    for k in 1..=max_opt {
        let mut cur = k;
        if k >= ans {
            break;
        }
        for i in 0..n {
            let r = if vat[i] % k == 0 { 0 } else { 1 };
            cur += 0.max(vat[i] / k + r - bucket[i]);
        }
        ans = ans.min(cur);
    }
    ans
}
// LCP 40: https://leetcode.cn/problems/uOAnQW/
pub fn maxmium_score(mut cards: Vec<i32>, cnt: i32) -> i32 {
    cards.sort();
    let cnt = cnt as usize;
    let former = &cards[0..cards.len() - cnt];
    let latter = &cards[cards.len() - cnt..];

    let sum = latter.iter().sum::<i32>();
    if sum % 2 == 0 {
        return sum;
    }
    let (mut min_odd, mut max_odd) = (-1, -1);
    let (mut min_even, mut max_even) = (-1, -1);

    for &ele in former.iter().rev() {
        if max_odd > 0 && max_even > 0 {
            break;
        }
        if ele % 2 == 0 {
            if max_even == -1 {
                max_even = ele;
            }
        } else if max_odd == -1 {
            max_odd = ele;
        }
    }
    for &ele in latter {
        if min_even > 0 && min_odd > 0 {
            break;
        }
        if ele % 2 == 0 {
            if min_even == -1 {
                min_even = ele;
            }
        } else if min_odd == -1 {
            min_odd = ele;
        }
    }
    let mut new_sum = 0;
    if min_odd != -1 && max_even != -1 {
        new_sum = sum + max_even - min_odd;
    }
    if min_even != -1 && max_odd != -1 {
        new_sum = new_sum.max(sum + max_odd - min_even);
    }
    new_sum
}
// LCS 01: https://leetcode.cn/problems/Ju9Xwi/
pub fn least_minutes(n: i32) -> i32 {
    let mut res = n;
    let mut speed = 1;
    let mut upgrade_time = 0;
    let mut download_time;
    while speed <= n {
        speed <<= 1;
        upgrade_time += 1;
        download_time = n / speed;
        if n % speed != 0 {
            download_time += 1;
        }
        res = res.min(upgrade_time + download_time);
    }
    res
}
// LCS 02: https://leetcode.cn/problems/WqXACV/
pub fn half_questions(questions: Vec<i32>) -> i32 {
    use std::collections::HashMap;
    let mut arr: Vec<i32> = questions
        .iter()
        .fold(HashMap::new(), |mut m, x| {
            *m.entry(x).or_insert(0) += 1;
            m
        })
        .iter()
        .map(|(_, &v)| v)
        .collect();
    arr.sort_unstable_by(|a, b| b.cmp(&a));
    let mut sum = 0;
    let mut kind = 0;
    for ele in arr {
        sum += ele as usize;
        kind += 1;
        if sum >= questions.len() / 2 {
            return kind;
        }
    }
    kind
}
// 剑指 Offer II 019: https://leetcode.cn/problems/RQku0D/, the same as 680
