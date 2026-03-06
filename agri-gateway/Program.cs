using Microsoft.AspNetCore.OpenApi;

var builder = WebApplication.CreateBuilder(args);

// ----------------------------------------------------
// ADD SERVICES
// ----------------------------------------------------
builder.Services.AddControllers();     // Enable Controllers
builder.Services.AddHttpClient();      // For calling Flask backend
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// ----------------------------------------------------
// MIDDLEWARE PIPELINE
// ----------------------------------------------------
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

// ----------------------------------------------------
// MAP CONTROLLERS
// ----------------------------------------------------
app.MapControllers();

app.Run();
