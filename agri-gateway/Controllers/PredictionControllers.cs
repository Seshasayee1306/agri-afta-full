using Microsoft.AspNetCore.Mvc;
using System.Net.Http.Json;

namespace AgriGateway.Controllers
{
    [ApiController]
    [Route("api")]
    public class PredictionController : ControllerBase
    {
        private readonly HttpClient _http;

        public PredictionController(HttpClient http)
        {
            _http = http;
        }

        [HttpPost("predict")]
        public async Task<IActionResult> Predict([FromBody] object payload)
        {
            var response = await _http.PostAsJsonAsync(
                "http://localhost:8001/predict", payload);

            var result = await response.Content.ReadAsStringAsync();
            return Content(result, "application/json");
        }

        [HttpPost("explain")]
        public async Task<IActionResult> Explain([FromBody] object payload)
        {
            var response = await _http.PostAsJsonAsync(
                "http://localhost:8001/explain", payload);

            var result = await response.Content.ReadAsStringAsync();
            return Content(result, "application/json");
        }
    }
}

