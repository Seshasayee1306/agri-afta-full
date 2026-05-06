using Microsoft.AspNetCore.Mvc;
using System.Net.Http.Json;
using System.Net.Http.Headers;

namespace AgriGateway.Controllers
{
    [ApiController]
    [Route("api")]
    public class PredictionController : ControllerBase
    {
        private readonly HttpClient _http;
        private const string BackendBase = "http://agri-backend:8000";
        private const string DiseaseBase = "http://disease-service:8010";

        public PredictionController(HttpClient http)
        {
            _http = http;
        }

        private async Task<IActionResult> ProxyJson(HttpMethod method, string url, object? payload = null)
        {
            using var request = new HttpRequestMessage(method, url);
            if (payload is not null)
            {
                request.Content = JsonContent.Create(payload);
            }

            using var response = await _http.SendAsync(request);
            var body = await response.Content.ReadAsStringAsync();
            var contentType = response.Content.Headers.ContentType?.ToString() ?? "application/json";
            return new ContentResult
            {
                StatusCode = (int)response.StatusCode,
                ContentType = contentType,
                Content = body
            };
        }

        [HttpPost("predict")]
        public Task<IActionResult> Predict([FromBody] object payload) =>
            ProxyJson(HttpMethod.Post, $"{BackendBase}/predict", payload);

        [HttpPost("predict_full_intelligent")]
        public Task<IActionResult> PredictFullIntelligent([FromBody] object payload) =>
            ProxyJson(HttpMethod.Post, $"{BackendBase}/predict_full_intelligent", payload);

        [HttpPost("explain")]
        public Task<IActionResult> Explain([FromBody] object payload) =>
            ProxyJson(HttpMethod.Post, $"{BackendBase}/explain", payload);

        [HttpPost("label")]
        public Task<IActionResult> Label([FromBody] object payload) =>
            ProxyJson(HttpMethod.Post, $"{BackendBase}/label", payload);

        [HttpGet("sensor_readings/latest")]
        public Task<IActionResult> GetLatestSensorReadings() =>
            ProxyJson(HttpMethod.Get, $"{BackendBase}/sensor_readings/latest");

        [HttpPost("sensor_readings")]
        public Task<IActionResult> IngestSensorReadings([FromBody] object payload) =>
            ProxyJson(HttpMethod.Post, $"{BackendBase}/sensor_readings", payload);

        [HttpPost("predict_disease")]
        public async Task<IActionResult> PredictDisease()
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, $"{DiseaseBase}/predict_disease");

            if (Request.HasFormContentType)
            {
                var form = await Request.ReadFormAsync();
                var multipart = new MultipartFormDataContent();

                foreach (var field in form)
                {
                    multipart.Add(new StringContent(field.Value.ToString()), field.Key);
                }

                foreach (var file in form.Files)
                {
                    await using var fileStream = file.OpenReadStream();
                    using var ms = new MemoryStream();
                    await fileStream.CopyToAsync(ms);
                    var bytes = ms.ToArray();

                    var fileContent = new ByteArrayContent(bytes);
                    if (!string.IsNullOrWhiteSpace(file.ContentType))
                    {
                        fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse(file.ContentType);
                    }

                    multipart.Add(fileContent, file.Name, file.FileName);
                }

                request.Content = multipart;
            }
            else
            {
                request.Content = new StreamContent(Request.Body);
                if (!string.IsNullOrWhiteSpace(Request.ContentType))
                {
                    request.Content.Headers.ContentType = MediaTypeHeaderValue.Parse(Request.ContentType);
                }
            }

            using var response = await _http.SendAsync(request);
            var body = await response.Content.ReadAsStringAsync();
            var contentType = response.Content.Headers.ContentType?.ToString() ?? "application/json";

            return new ContentResult
            {
                StatusCode = (int)response.StatusCode,
                ContentType = contentType,
                Content = body
            };
        }
    }
}
