//Created on Mai 4 10:56:16 2026
//@author: kambana

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const availability = new Rate('availability');
const errorRate = new Rate('error_rate');
const latencyTrend = new Trend('latency');

export const options = {
  vus: 1,
  duration: '6m',

  insecureSkipTLSVerify: true,

  thresholds: {
    availability: ['rate>0.99'],
    error_rate: ['rate<0.01'],
    latency: ['p(95)<500'],
  },
};

export default function () {
  const url = 'http://IP_ADDRESS_OF_DT_NODE_RED_APPLICATION:1880/';

  const res = http.get(url, {
    timeout: '5s',
  });

  const success = check(res, {
    'status is valid (2xx-3xx)': (r) =>
      r.status >= 200 && r.status < 400,
  });

  availability.add(success);
  errorRate.add(!success);

  const latency = res.timings.duration;
  latencyTrend.add(latency);

  if (!success) {
    console.log(`ERROR -> status=${res.status}`);
  }

  sleep(1);
}